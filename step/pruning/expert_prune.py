from tqdm import tqdm
from copy import deepcopy
import math
import logging
import torch
from torch import nn
import torch.nn.functional as F
# from transformers import DeepseekV3MLP
# from step.model.deepseek_v2.modeling_deepseek import DeepseekV2MLP
# from step.model.moonshotaiv2.modeling_deepseek import DeepseekV3MLP
from step.model.olmoe.modeling_olmoe import OlmoeMLP
# from step.model.deepseek.modeling_deepseek import DeepseekMLP
# from step.model.qwen3.modeling_qwen3_moe import Qwen3MoeMLP
# from step.model.glm4.modeling_glm4_moe import Glm4MoeMLP
# from step.model.bailing.modeling_bailing_moe_v2 import BailingMoeV2MLP
@torch.no_grad()
def expert_prune_by_routing_score(args, model, train_dataloader):
    # Move the model to the GPU 
    #model.cuda()
    # Retrieves the index of the currently active GPU device
    device = torch.cuda.current_device()

    handles = []
    scores, denominator = {}, {}

    # Get MoE model info
    num_layers = model.config.num_hidden_layers
    if "deepseek" in model.config.model_type:
        num_experts = model.config.n_routed_experts
    elif "olmoe" in model.config.model_type:
        num_experts = model.config.num_experts
    elif "qwen" in model.config.model_type:
        num_experts = model.config.num_experts
    elif "ling" in model.config.model_type:
        num_experts = model.config.num_experts
    else:
        raise ValueError(f"unknow model type: {model.config.model_type}")
    
    # Identify MoE layer
    if "deepseek" in model.config.model_type:
        valid_moe_layer_indices = [
            layer_idx for layer_idx in range(num_layers) 
            if (
                model.config.n_routed_experts is not None and
                layer_idx >= model.config.first_k_dense_replace and 
                layer_idx % model.config.moe_layer_freq == 0
            )
        ]
    elif "olmoe" in model.config.model_type:
        valid_moe_layer_indices = list(range(num_layers))
    elif "qwen" in model.config.model_type:
        valid_moe_layer_indices = list(range(num_layers))
    elif "ling" in model.config.model_type:
        valid_moe_layer_indices = [
            layer_idx for layer_idx in range(num_layers) 
            if (model.config.num_experts is not None and layer_idx >= model.config.first_k_dense_replace)
        ]
    # Register forward hooks
    for i in valid_moe_layer_indices:
        layer = model.model.layers[i] # DeepseekV2DecoderLayer with MoE layer and number of experts > preserve_n_experts

        def create_hook(layer_idx):
            def stateful_hook(module, _input, _output):
                batch_size = _input[0].shape[0]

       
                if 'olmoe' in model.config.model_type:
                    router_logits = _output
                    topk_weight = F.softmax(router_logits, dim=1, dtype=torch.float)
                    topk_weight, topk_idx = torch.topk(topk_weight, model.config.num_experts_per_tok, dim=-1)
              
                routing_weights = torch.zeros(
                    (topk_weight.shape[0], num_experts),
                    device=topk_weight.device,
                    dtype=torch.float
                )
                routing_weights = torch.scatter(routing_weights, dim=1, index=topk_idx, src=topk_weight.to(torch.float))

                if layer_idx not in scores:
                    denominator[layer_idx] = batch_size
                    scores[layer_idx] = routing_weights.float().sum(0)
                else:
                    denominator[layer_idx] += batch_size
                    scores[layer_idx] += routing_weights.float().sum(0)

            return stateful_hook

        # Get MoE gate
        moe_gate_layer = layer.mlp.gate

        # register forward hook
        handle = moe_gate_layer.register_forward_hook(create_hook(i))
        handles.append(handle)

    # Execute model
    data_iter = iter(train_dataloader)
    for step in tqdm(range(args.max_steps), desc="collecting accumulated routing scores"):
        batch = next(data_iter)
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)
    
    # clear handles before saving
    for handle in handles:
        handle.remove()

    # save the pruned model state, this should not introduce more GPU memory usage
    #model.cpu()
    state_dict = model.state_dict()

    experts_to_keep_idx_dict = {}
    if args.expert_ranking_scope == 'layer':
        for layer_idx in scores.keys():
            # Calculate mean score
            score = scores[layer_idx] / denominator[layer_idx]
            
            # Get topK experts 
            _, experts_to_keep_idx = torch.topk(
                score,
                args.preserve_n_experts,
                largest=True
            )
            experts_to_keep_idx_dict[layer_idx] = sorted(experts_to_keep_idx.tolist())
    else:
        metric = torch.cat(list(scores.values()))
        sorted_scores, _ = torch.sort(metric, descending=True)
        threshold = sorted_scores[math.ceil(len(metric)*args.preserve_n_experts/num_experts)]
        for layer_idx in scores.keys():
            experts_to_keep_idx_dict[layer_idx] = sorted((torch.where(scores[layer_idx]>threshold)[0]).tolist())
    
    class zero_module(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
        def forward(self,inp):
            return torch.zeros_like(inp)
        
    new_routed_experts = {}
    for layer_idx in valid_moe_layer_indices:
        experts_to_keep_idx = experts_to_keep_idx_dict[layer_idx]
        if len(experts_to_keep_idx) == 0:
            logging.warn(f"experts of layer {layer_idx} should have been fully removed. We preserve one for compatibility")
            experts_to_keep_idx.append(torch.argmax(scores[layer_idx]).item())
        new_routed_experts[layer_idx] = len(experts_to_keep_idx)
        
        if "deepseek" in model.config.model_type:
            num_experts = model.config.n_routed_experts 
        elif "olmoe" in model.config.model_type:
            num_experts = model.config.num_experts 
        elif "qwen" in model.config.model_type:
            num_experts = model.config.num_experts 
        elif "ling" in model.config.model_type:
            num_experts = model.config.num_experts 
        # Remove pruned experts
        ffn = model.model.layers[layer_idx].mlp
        
        # gate_proj_weight = torch.zeros_like(ffn.experts[0].gate_proj.weight)
        # down_proj_weight = torch.zeros_like(ffn.experts[0].down_proj.weight)
        # up_proj_weight = torch.zeros_like(ffn.experts[0].up_proj.weight)
        for i in range(num_experts):
            if i not in experts_to_keep_idx:
                #ffn.experts[i]=zero_module()
                if args.save_model:
                    ffn.experts[i].gate_proj.weight.data = torch.zeros_like(ffn.experts[0].gate_proj.weight.data)
                    ffn.experts[i].down_proj.weight.data = torch.zeros_like(ffn.experts[0].down_proj.weight.data)
                    ffn.experts[i].up_proj.weight.data = torch.zeros_like(ffn.experts[0].up_proj.weight.data)
                else:
                    ffn.experts[i]=zero_module()


    return model


@torch.no_grad()
def align_expert_weight(reference_mlp, target_mlp):
    from scipy.optimize import linear_sum_assignment

    lsa_cost_matrix = torch.mm(
        reference_mlp.gate_proj.weight.data.float(), target_mlp.gate_proj.weight.data.float().t()
    )
    lsa_cost_matrix += torch.mm(
        reference_mlp.up_proj.weight.data.float(), target_mlp.up_proj.weight.data.float().t()
    )
    lsa_cost_matrix += torch.mm(
        reference_mlp.down_proj.weight.data.float().t(), target_mlp.down_proj.weight.data.float()
    )
    _, perm = linear_sum_assignment(lsa_cost_matrix.cpu().numpy(), maximize=True)


    d_ff = target_mlp.gate_proj.out_features

    # Check the permutation vector
    if perm.shape != (d_ff,):
        raise ValueError(f"The shape of the permutation vector should be (d_ff, ), but got {perm.shape}.")

    # Permute the weights of the MLP
    target_mlp.gate_proj.weight.data = target_mlp.gate_proj.weight.data[perm, :]
    target_mlp.up_proj.weight.data = target_mlp.up_proj.weight.data[perm, :]
    target_mlp.down_proj.weight.data = target_mlp.down_proj.weight.data[:, perm]

    return target_mlp


@torch.no_grad()
def expert_prune_by_mc_smoe(args, model, train_dataloader):
    # Move the model to the GPU 
    #model.cuda()
    # Retrieves the index of the currently active GPU device
    device = torch.cuda.current_device()

    handles = []
    # Get MoE model info
    num_layers = model.config.num_hidden_layers
    if "deepseek" in model.config.model_type:
        num_experts = model.config.n_routed_experts
    elif "olmoe" in model.config.model_type:
        num_experts = model.config.num_experts
    elif "qwen" in model.config.model_type:
        num_experts = model.config.num_experts
    elif "glm" in model.config.model_type:
        num_experts = model.config.n_routed_experts
    else:
        raise ValueError(f"unknow model type: {model.config.model_type}")
    
    # Identify MoE layer
    if "deepseek" in model.config.model_type:
        valid_moe_layer_indices = [
            layer_idx for layer_idx in range(num_layers) 
            if (
                model.config.n_routed_experts is not None and
                layer_idx >= model.config.first_k_dense_replace and 
                layer_idx % model.config.moe_layer_freq == 0
            )
        ]
    elif "olmoe" in model.config.model_type:
        valid_moe_layer_indices = list(range(num_layers))
    elif "qwen" in model.config.model_type:    
        valid_moe_layer_indices = list(range(num_layers))
    # step 1: mlp weight permutation to align expert weight channels
    for i in tqdm(valid_moe_layer_indices, desc="Expert weight permuation"):
        layer = model.model.layers[i]
        for expert_idx in range(1, num_experts):
            layer.mlp.experts[expert_idx] = align_expert_weight(
                layer.mlp.experts[0], 
                layer.mlp.experts[expert_idx],
            )
    
    # step 2: merge experts according to access frequency and activation similarity
    sim_matrix = {}
    access_frequency = {}
    for i in valid_moe_layer_indices:
        layer = model.model.layers[i]
        sim_matrix[i] = torch.zeros(
            num_experts, num_experts, device=device, dtype=torch.float
        ) + torch.eye(num_experts, device=device, dtype=torch.float)
        access_frequency[i] = torch.zeros(
            num_experts, device=device, dtype=torch.float
        )
        def create_similarity_and_usage_hook(layer_idx):
            def stateful_hook(module, _input, _output):
                # _compute_all_similarities_by_router_logits
                hidden_states = _input[0]
                hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
                # bs*seq_len, num_expert
                scores = F.sigmoid(
                    F.linear(hidden_states.float(), module.weight.float())
                )
                for e_i in range(num_experts-1):
                    for e_j in range(e_i+1, num_experts):
                        weight_i = scores[:, e_i].flatten()
                        weight_j = scores[:, e_j].flatten()
                        sim_matrix[layer_idx][e_i, e_j] = (F.cosine_similarity(
                            weight_i, weight_j, dim=-1, eps=1e-7
                        ) + 1) / 2

                # compute_all_usages
                
                if 'olmoe' in model.config.model_type:
                    router_logits = _output
                    topk_weight = F.softmax(router_logits, dim=1, dtype=torch.float)
                    _, topk_idx = torch.topk(topk_weight, model.config.num_experts_per_tok, dim=-1)
                    topk_indices = topk_idx.view(-1)
                
                    
                access_frequency[layer_idx].scatter_add_(0, topk_indices, torch.ones_like(topk_indices, dtype=torch.float))
                access_frequency[layer_idx] = access_frequency[layer_idx] / torch.sum(access_frequency[layer_idx])

            return stateful_hook
        
        handle = layer.mlp.gate.register_forward_hook(create_similarity_and_usage_hook(i))
        handles.append(handle)

    # update sim_matrix and access_frequency
    data_iter = iter(train_dataloader)
    for step in tqdm(range(args.max_steps), desc="collecting similarities"):
        batch = next(data_iter)
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)

    # group_experts_into_clusters_by_routing_guided_globally
    # _assign_num_groups_per_layer
    total_num_groups = args.preserve_n_experts * len(valid_moe_layer_indices)
    all_usage_frequency = []
    usage_frequency_dict = deepcopy(access_frequency)
    for i in valid_moe_layer_indices:
        print(f">>> layer {i}:\naccess_frequency:\n{access_frequency[i]}\nsim_matrix:\n{sim_matrix[i]}")
        max_usage_index = torch.argmax(usage_frequency_dict[i])
        usage_frequency_dict[i][max_usage_index] = 1.0
        all_usage_frequency.append(usage_frequency_dict[i])

    all_usage_frequency = torch.cat(all_usage_frequency, dim=0)
    sorted_usage_frequency, sorted_indices = torch.sort(all_usage_frequency, descending=True)
    frequency_threshold = sorted_usage_frequency[total_num_groups]

    num_groups_per_layer = dict()
    for i in valid_moe_layer_indices:
        num_groups_per_layer[i] = torch.sum(
            (usage_frequency_dict[i]>frequency_threshold).long()
        ).item()
        print(f">>> layer {i} group: {num_groups_per_layer[i]}")
    
    core_experts = dict()
    group_state_dict = dict()
    for i in tqdm(valid_moe_layer_indices, desc="grouping experts layer by layer"):
        group_state_dict[i] = torch.arange(num_experts, device=device)
        # Assign top-K most-used experts with label 0 to K-1 respectively
        num_groups = num_groups_per_layer[i]
        group_member_count = torch.zeros(num_groups)
        indices_sorted_by_usage = torch.argsort(access_frequency[i], descending=True)
        core_expert_indices = indices_sorted_by_usage[:num_groups]
        core_experts[i] = core_expert_indices.tolist()
        for g_idx in range(num_groups):
            group_member_count[g_idx] += 1
            group_state_dict[i][core_expert_indices[g_idx]] = g_idx
        
        similarity_matrix = sim_matrix[i]
        for r_idx in range(num_groups, num_experts):
            expert_idx = indices_sorted_by_usage[r_idx]
            most_similar_core = core_expert_indices[
                torch.argmax(similarity_matrix[expert_idx, core_expert_indices])
            ]
            most_similar_group_label = group_state_dict[i][most_similar_core]
            group_state_dict[i][expert_idx] = most_similar_group_label
            group_member_count[most_similar_group_label] += 1
            if group_member_count[group_state_dict[i][expert_idx]] >= num_experts:
                raise ValueError(
                    f"group_member_count[group_state_dict[i][expert_idx]]={group_member_count[group_state_dict[i][expert_idx]]} >= num_experts={num_experts}")
        
        print(f"layer {i} group_member_count: {group_member_count}")
    
    # merge_by_groups_with_usage_frequency_weighting
    # TODO: step 3: compress experts
    new_num_experts = {}
    for i in tqdm(valid_moe_layer_indices, desc="merging experts layer by layer"):
        group_labels = group_state_dict[i]
        usage_frequencies = usage_frequency_dict[i]
        mlp = model.model.layers[i].mlp
        new_weights = []
        for label in group_labels.unique():
            expert_indices = torch.where(group_labels == label)[0]
            gate_proj_weight_list = torch.stack([
                mlp.experts[expert_idx].gate_proj.weight * usage_frequencies[expert_idx] \
                    for expert_idx in expert_indices
            ], dim=0)
            gate_proj_weight = torch.sum(gate_proj_weight_list, dim=0)/(
                torch.sum(usage_frequencies[expert_indices], dim=0) + 1e-7
            )

            up_proj_weight_list = torch.stack([
                mlp.experts[expert_idx].up_proj.weight * usage_frequencies[expert_idx] \
                    for expert_idx in expert_indices
            ], dim=0)
            up_proj_weight = torch.sum(up_proj_weight_list, dim=0) / (
                torch.sum(usage_frequencies[expert_indices], dim=0) + 1e-7
            )

            down_proj_weight_list = torch.stack([
                mlp.experts[expert_idx].down_proj.weight * usage_frequencies[expert_idx] \
                    for expert_idx in expert_indices
            ], dim=0)
            down_proj_weight = torch.sum(down_proj_weight_list, dim=0) / (
                torch.sum(usage_frequencies[expert_indices], dim=0) + 1e-7
            )
            
            for e_idx in expert_indices:
                mlp.experts[e_idx].gate_proj.weight.copy_(gate_proj_weight)
                mlp.experts[e_idx].up_proj.weight.copy_(up_proj_weight)
                mlp.experts[e_idx].down_proj.weight.copy_(down_proj_weight)

    # clear handles before saving
    for handle in handles:
        handle.remove()

    return model


@torch.no_grad()
def expert_prune_by_mone(args, model, train_dataloader):
    # Move the model to the GPU 
    # #model.cuda()
    # Retrieves the index of the currently active GPU device
    device = torch.cuda.current_device()

    handles = []
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    # if "deepseek_v3" in model.config.model_type:
    #     novice_cls = DeepseekV3MLP
    #     num_experts = model.config.n_routed_experts
    #     intermediate_size = model.config.moe_intermediate_size
    # elif "deepseek_v2" in model.config.model_type:
    #     novice_cls = DeepseekV2MLP
    #     num_experts = model.config.n_routed_experts
    #     intermediate_size = model.config.moe_intermediate_size
    if "olmoe" in model.config.model_type:
        mlp_module_class = OlmoeMLP
        num_experts = model.config.num_experts
        intermediate_size = model.config.intermediate_size
    # elif "deepseek" in model.config.model_type:
    #     novice_cls = DeepseekMLP
    #     num_experts = model.config.n_routed_experts
    #     intermediate_size = model.config.moe_intermediate_size
    # elif "qwen" in model.config.model_type:
    #     novice_cls = Qwen3MoeMLP
    #     num_experts = model.config.num_experts
    #     intermediate_size = model.config.moe_intermediate_size
    # elif "glm" in model.config.model_type:
    #     novice_cls = Glm4MoeMLP
    #     num_experts = model.config.n_routed_experts
    #     intermediate_size = model.config.moe_intermediate_size
    # elif "ling" in model.config.model_type:
    #     novice_cls = BailingMoeV2MLP
    #     num_experts = model.config.num_experts
    #     intermediate_size = model.config.moe_intermediate_size
    else:
        raise ValueError(f"unknow model type: {model.config.model_type}")

    # Identify MoE layer
    if "deepseek" in model.config.model_type:
        valid_moe_layer_indices = [
            layer_idx for layer_idx in range(num_layers) 
            if (
                model.config.n_routed_experts is not None and
                layer_idx >= model.config.first_k_dense_replace and 
                layer_idx % model.config.moe_layer_freq == 0
            )
        ]
    elif "olmoe" in model.config.model_type:
        valid_moe_layer_indices = list(range(num_layers))
    elif "qwen" in model.config.model_type:
        valid_moe_layer_indices = [layer_idx for layer_idx in range(num_layers) if (layer_idx not in model.config.mlp_only_layers) and (
            model.config.num_experts > 0 and (layer_idx + 1) % model.config.decoder_sparse_step == 0
        )]
    elif "glm" in model.config.model_type:
        valid_moe_layer_indices = [
            layer_idx for layer_idx in range(num_layers) 
            if (
                model.config.n_routed_experts is not None and
                layer_idx >= model.config.first_k_dense_replace 
            )
        ]
    elif "ling" in model.config.model_type:
        valid_moe_layer_indices = [
            layer_idx for layer_idx in range(num_layers) 
            if (model.config.num_experts is not None and layer_idx >= model.config.first_k_dense_replace)
        ]
    #########################################
    # Create hooks to collect pruning metrics
    bias_stats = {}
    def create_expert_hook(expert_name):
        def stateful_expert_hook(module, _input, _output):
            out = _output
            out = out.view(-1, out.shape[-1])
            inp = _input[0]
            inp = inp.view(-1, inp.shape[-1])
            token_size = out.shape[0]

            # retrieve stats
            num_tokens = bias_stats[expert_name]["num_tokens"]
            baseline_out = bias_stats[expert_name]["baseline_out"]
            if args.expert_ranking_metric in ['routing_weighted_norm', 'io_fluctuation', 'fusion']:
                fluc_out = bias_stats[expert_name]["routing_weighted_norm"]
            elif args.expert_ranking_metric in ['token_fluctuation', 'intermediate_fluctuation', 'rs_intermediate']:
                baseline_inp = bias_stats[expert_name]["baseline_inp"]
                fluc_inp = bias_stats[expert_name]["input_feature_norm"]

            # update moving average and fluctuation
            if num_tokens > 0:
                baseline_out *= num_tokens / (num_tokens + token_size)
                baseline_out += torch.sum(out.float(), dim=0) / (num_tokens + token_size)
                if args.expert_ranking_metric in ['routing_weighted_norm', 'fusion']:
                    if num_tokens > 0:
                        try:
                            fluc_out *= (num_tokens - 1) / (num_tokens + token_size - 1)
                            fluc_out += torch.sum((out - baseline_out.unsqueeze(0)).float().pow(2), dim=0) / (num_tokens + token_size)
                        except:
                            pass
                elif args.expert_ranking_metric in ['io_fluctuation',]:
                    if num_tokens > 0:
                        fluc_out *= (num_tokens - 1) / (num_tokens + token_size - 1)
                        inp = _input[0]
                        inp = inp.view(-1, inp.shape[-1])
                        fluc_out += torch.sum((inp - out).float().pow(2), dim=0) / (num_tokens + token_size)
                elif args.expert_ranking_metric in ['token_fluctuation', 'intermediate_fluctuation', 'rs_intermediate']:
                    baseline_inp *= num_tokens / (num_tokens + token_size)
                    baseline_inp += torch.sum(inp, dim=0) / (num_tokens + token_size)
                    if num_tokens > 0:
                        fluc_inp *= (num_tokens - 1) / (num_tokens + token_size - 1)
                        fluc_inp += torch.sum((inp - baseline_inp.unsqueeze(0))**2, dim=0) / (num_tokens + token_size)
            
            # write back stats
            bias_stats[expert_name]["num_tokens"] += token_size
            bias_stats[expert_name]["baseline_out"] = baseline_out
            if args.expert_ranking_metric in ['routing_weighted_norm', 'io_fluctuation', 'fusion']:
                bias_stats[expert_name]['routing_weighted_norm'] = fluc_out
            elif args.expert_ranking_metric in ['token_fluctuation', 'intermediate_fluctuation', 'rs_intermediate']:
                bias_stats[expert_name]["baseline_inp"] = baseline_inp
                bias_stats[expert_name]["input_feature_norm"] = fluc_inp
        
        return stateful_expert_hook

    if args.expert_ranking_metric in ['routing_score', 'fusion', 'rs_intermediate']:
        routing_stats = {}
        def create_gate_hook(layer_idx):
            def stateful_gate_hook(module, _input, _output):
                batch_size = _input[0].shape[0]
                
                if 'olmoe' in model.config.model_type:
                    router_logits = _output
                    topk_weight = F.softmax(router_logits, dim=1, dtype=torch.float)
                    topk_weight, topk_idx = torch.topk(topk_weight, model.config.num_experts_per_tok, dim=-1)
                
                    
                assert topk_idx.dim() == 2
                # token_size = topk_idx.shape[0]

                routing_weights = torch.zeros(
                    (topk_weight.shape[0], num_experts),
                    device=topk_weight.device, dtype=torch.float
                )
                # num_tokens_per_expert = torch.zeros_like(routing_weights)

                routing_weights = torch.scatter(routing_weights, dim=1, index=topk_idx, src=topk_weight.to(torch.float))
                # num_tokens_per_expert = torch.scatter_add(num_tokens_per_expert, dim=1, index=topk_idx, src=torch.ones_like(topk_weight))
                # num_tokens_per_expert = torch.sum(num_tokens_per_expert, dim=0)

                scores = routing_stats[layer_idx]["scores"]
                num_tokens = routing_stats[layer_idx]["num_tokens"]
                
                # scores *= num_tokens / (num_tokens + num_tokens_per_expert)
                # scores += torch.sum(routing_weights, dim=0) / (num_tokens + num_tokens_per_expert)
                
                scores *= num_tokens / (num_tokens + batch_size)
                scores += torch.sum(routing_weights, dim=0) / (num_tokens + batch_size)

                routing_stats[layer_idx]["num_tokens"] += batch_size #num_tokens_per_expert
                routing_stats[layer_idx]["scores"] = scores

            return stateful_gate_hook
    
        layer_attn_mask={}
        # def create_attn_hook(layer_idx):
        #     def stateful_attn_hook(module, _input, _output):
        #         #batch_size = _input[0].shape[0]
        #         if 'deepseek' in model.config.model_type:
        #             attn_output, attn_weights = _output[:2]
        #             delete_num = int(attn_weights.shape[3]*args.tau)
        #             attn_self_attention = attn_weights.max(dim=1).values.squeeze()
        #             attn_self_attention = attn_self_attention.sum(dim=0)
        #             topk = torch.sort(attn_self_attention * attn_self_attention, dim=-1).indices[:delete_num]
        #             topk_idx = torch.ones(attn_weights.shape[3], dtype=torch.bool, device=attn_weights.device)
        #             topk_idx[topk] = False
        #             layer_attn_mask[layer_idx]=topk_idx
        #         elif 'olmoe' in model.config.model_type:
        #             router_logits = _output
        #             topk_weight = F.softmax(router_logits, dim=1, dtype=torch.float)
        #             topk_weight, topk_idx = torch.topk(topk_weight, model.config.num_experts_per_tok, dim=-1)

        #     return stateful_attn_hook

        
    for i in valid_moe_layer_indices:
        layer = model.model.layers[i]
        mlp = model.model.layers[i].mlp
        if args.expert_ranking_metric in ['routing_score', 'fusion','rs_intermediate']:
            routing_stats[i] = {
                "scores": torch.zeros(num_experts, device=next(mlp.parameters()).device, dtype=torch.float),
                "num_tokens": torch.zeros(num_experts, device=next(mlp.parameters()).device, dtype=torch.float),
            }
            
            # handle = layer.self_attn.register_forward_hook(create_attn_hook(i))
            # handles.append(handle)

            handle = mlp.gate.register_forward_hook(create_gate_hook(i))
            handles.append(handle)
            
            

        for e_idx in range(len(mlp.experts)):
            expert_name = f"layers.{i}.experts.{e_idx}"
            bias_stats[expert_name] = {
                "num_tokens": 0,
                "baseline_out": torch.zeros(hidden_size, device=next(mlp.parameters()).device, dtype=torch.float),
            }

            if args.expert_ranking_metric in ['routing_weighted_norm', 'io_fluctuation', 'fusion']:
                bias_stats[expert_name]['routing_weighted_norm'] = torch.zeros(hidden_size, device=next(mlp.parameters()).device, dtype=torch.float)
                handle = mlp.experts[e_idx].register_forward_hook(create_expert_hook(expert_name))
                handles.append(handle)
            elif args.expert_ranking_metric in ['token_fluctuation']:
                bias_stats[expert_name]["baseline_inp"] = torch.zeros(hidden_size, device=next(mlp.parameters()).device, dtype=torch.float)
                bias_stats[expert_name]["input_feature_norm"] = torch.zeros(hidden_size, device=next(mlp.parameters()).device, dtype=torch.float)
                handle = mlp.experts[e_idx].register_forward_hook(create_expert_hook(expert_name))
                handles.append(handle)
            elif args.expert_ranking_metric in ['intermediate_fluctuation']:
                bias_stats[expert_name] = {
                    "num_tokens": 0,
                    "baseline_out": torch.zeros(hidden_size, device=next(mlp.parameters()).device, dtype=torch.float),
                    "baseline_inp": torch.zeros(intermediate_size, device=next(mlp.parameters()).device, dtype=torch.float),
                    "input_feature_norm": torch.zeros(intermediate_size, device=next(mlp.parameters()).device, dtype=torch.float),
                }
                handle = mlp.experts[e_idx].down_proj.register_forward_hook(create_expert_hook(expert_name))
                handles.append(handle)

    data_iter = iter(train_dataloader)
    for step in tqdm(range(args.max_steps), desc="collecting accumulated stats"):
        batch = next(data_iter)
        with torch.no_grad():
            batch = {k: v.to("cuda:0") for k, v in batch.items()}
            model(**batch)
    
    for handle in handles:
        handle.remove()
    
    #########################
    # Collect pruning metrics
    metric_list = {}
    if args.expert_ranking_metric == 'routing_score':
        for layer_idx in valid_moe_layer_indices:
            metric_list[layer_idx] = routing_stats[layer_idx]["scores"]
    elif args.expert_ranking_metric in ['routing_weighted_norm', 'io_fluctuation']:
        for layer_idx in valid_moe_layer_indices:
            fluc_list = [bias_stats[f'layers.{layer_idx}.experts.{e_idx}']['routing_weighted_norm'] for e_idx in range(num_experts)]
            output_fluc = torch.stack(fluc_list)
            metric_list[layer_idx] = torch.norm(output_fluc, dim=1)
    elif args.expert_ranking_metric == 'fusion':
        for layer_idx in valid_moe_layer_indices:
            fluc_list = [bias_stats[f'layers.{layer_idx}.experts.{e_idx}']['routing_weighted_norm'] for e_idx in range(num_experts)]
            # num_experts
            output_fluc = torch.norm(torch.sqrt(torch.stack(fluc_list)), dim=1)
            metric_list[layer_idx] = (args.fusion_io_weight * output_fluc) * ((1-args.fusion_io_weight) * routing_stats[layer_idx]["scores"])
    elif args.expert_ranking_metric=='token_fluctuation':
        for layer_idx in valid_moe_layer_indices:
            fluc_list = [bias_stats[f'layers.{layer_idx}.experts.{e_idx}']['input_feature_norm'] for e_idx in range(num_experts)]
            input_fluc = torch.stack(fluc_list)
            metric_list[layer_idx] = torch.norm(input_fluc, dim=1)
    elif args.expert_ranking_metric=='intermediate_fluctuation':
        for layer_idx in valid_moe_layer_indices:
            scores = []
            for e_idx in range(num_experts):
                inp_fluc = bias_stats[f'layers.{layer_idx}.experts.{e_idx}']['input_feature_norm']
                mlp = model.model.layers[layer_idx].mlp
                intermediate_score = inp_fluc * torch.sum(mlp.experts[e_idx].down_proj.weight.data.pow(2), dim=0)
                scores.append(intermediate_score)
            metric_list[layer_idx] = torch.norm(torch.stack(scores), dim=1)
    else:
        raise ValueError(f"unknow ranking metric: {args.expert_ranking_metric}")
    expert_sort=[]
    ###########################
    # Collect pruning threshold
    if args.expert_ranking_scope == 'model':
        #metric = torch.cat(list(metric_list.values()))
        metrics = [metric.to('cuda:0') for metric in metric_list.values()]
        metric = torch.cat(metrics)
        sorted_scores, _ = torch.sort(metric, descending=True)
        threshold_val = sorted_scores[math.ceil(len(sorted_scores)*args.preserve_n_experts/num_experts)]
        threshold = {layer_idx: threshold_val for layer_idx in valid_moe_layer_indices}
    else:
        threshold = {}
        for layer_idx in metric_list:
            metric = metric_list[layer_idx]
            sorted_scores, indices = torch.sort(metric, descending=True)
            expert_sort.append(indices.unsqueeze(0))
            threshold[layer_idx] = sorted_scores[args.preserve_n_experts]

    #################
    # Run pruning process
    approximate_experts = {}
    approximate_expert_init_tokens = {}
    for layer_idx in valid_moe_layer_indices:
        layer_metric = (metric_list[layer_idx]).to(threshold[layer_idx].device)
        expert_mask = layer_metric > threshold[layer_idx]

        expert_indicator_list = expert_mask.tolist()
        approximate_experts[layer_idx] = []
        approximate_expert_init_tokens[layer_idx] = []
        for expert_idx, is_preserved in enumerate(expert_indicator_list):
            if not is_preserved:
                
                approximate_experts[layer_idx].append(expert_idx)
                approximate_expert_init_tokens[layer_idx].append(
                    bias_stats[expert_name]['num_tokens'] if args.enable_novice_evolving else 0
                )
                expert_name = f"layers.{layer_idx}.experts.{expert_idx}"
                novice = novice_cls(model.config, is_approx=True, 
                    acc_tokens=bias_stats[expert_name]['num_tokens'] if args.enable_novice_evolving else 0)
                novice.approx_value.copy_(
                    bias_stats[expert_name]["baseline_out"]
                )                
                model.model.layers[layer_idx].mlp.experts[expert_idx] = novice.bfloat16().to(next(model.model.layers[layer_idx].mlp.parameters()).device)
    # update configs
    model.config.approximate_experts = approximate_experts
    model.config.approximate_expert_init_tokens = approximate_expert_init_tokens
    #model.cpu()
    torch.cuda.empty_cache()
    return model

@torch.no_grad()
def expert_prune_by_freq(args, model, train_dataloader):
    # Move the model to the GPU 
    #model.cuda()
    # Retrieves the index of the currently active GPU device
    device = torch.cuda.current_device()

    handles = []
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
   
    if "olmoe" in model.config.model_type:
        mlp_module_class = OlmoeMLP
        num_experts = model.config.num_experts
        intermediate_size = model.config.intermediate_size
    else:
        raise ValueError(f"unknow model type: {model.config.model_type}")

    # Identify MoE layer
    if "deepseek" in model.config.model_type:
        valid_moe_layer_indices = [
            layer_idx for layer_idx in range(num_layers) 
            if (
                model.config.n_routed_experts is not None and
                layer_idx >= model.config.first_k_dense_replace and 
                layer_idx % model.config.moe_layer_freq == 0
            )
        ]
    elif "olmoe" in model.config.model_type:
        valid_moe_layer_indices = list(range(num_layers))
    elif "qwen" in model.config.model_type:
        valid_moe_layer_indices = [layer_idx for layer_idx in range(num_layers) if (layer_idx not in model.config.mlp_only_layers) and (
            model.config.num_experts > 0 and (layer_idx + 1) % model.config.decoder_sparse_step == 0
        )]
    elif "glm" in model.config.model_type:
        valid_moe_layer_indices = [
            layer_idx for layer_idx in range(num_layers) 
            if (
                model.config.n_routed_experts is not None and
                layer_idx >= model.config.first_k_dense_replace 
            )
        ]
    elif "ling" in model.config.model_type:
        valid_moe_layer_indices = [
            layer_idx for layer_idx in range(num_layers) 
            if (model.config.num_experts is not None and layer_idx >= model.config.first_k_dense_replace)
        ]
    #########################################
    # Create hooks to collect pruning metrics
    def merge(t1,t2):
        # Create zero tensor with same length as t1
        result = torch.zeros_like(t1)

        # Find intersection positions between t1 and t2 (using broadcast and element-wise comparison)
        mask = (t1.unsqueeze(1) == t2).any(dim=1)  # shape: [m]

        # Set intersection positions to 1, keep others as 0
        result[mask] = 1
        
        return result
            
    bias_stats = {}
    layer_attn_mask = {}
    def create_expert_hook(expert_name):
        def stateful_expert_hook(module, _input, _output):
            out = _output
            out = out.view(-1, out.shape[-1])
            inp = _input[0]
            inp = inp.view(-1, inp.shape[-1])
            token_size = out.shape[0]
            layer_idx, expert_idx = int(expert_name.split('.')[1]), int(expert_name.split('.')[-1])
            tmp_rs=routing_stats[layer_idx]["topk_idx"].clone()
            tmp_rs[~layer_attn_mask[layer_idx]]=-1
            if not torch.all(tmp_rs==expert_idx):
                
                routing_idx=(routing_stats[layer_idx]["topk_idx"]==expert_idx).view(-1)
                tmp_routing_idx=torch.where((tmp_rs==expert_idx).view(-1))[0]//model.config.num_experts_per_tok
                tmp_routing_idx1=torch.where(routing_idx)[0]//model.config.num_experts_per_tok
                route_indx=merge(tmp_routing_idx1,tmp_routing_idx)
                output_routing=routing_stats[layer_idx]["topk_weight"].view(-1)[torch.where((tmp_rs==expert_idx).view(-1))[0]].unsqueeze(dim=-1)
                # retrieve stats
                num_tokens = bias_stats[expert_name]["num_tokens"]
                baseline_out = bias_stats[expert_name]["baseline_out"]
                if args.expert_ranking_metric in ['routing_weighted_norm', 'io_fluctuation', 'fusion']:
                    fluc_out = bias_stats[expert_name]["routing_weighted_norm"]
                elif args.expert_ranking_metric in ['token_fluctuation', 'intermediate_fluctuation', 'rs_intermediate']:
                    baseline_inp = bias_stats[expert_name]["baseline_inp"]
                    fluc_inp = bias_stats[expert_name]["input_feature_norm"]
                if num_tokens > 0:
                    # update moving average and fluctuation
                    baseline_out *= num_tokens / (num_tokens + token_size)
                    baseline_out += torch.sum(out.float(), dim=0) / (num_tokens + token_size)
                    if args.expert_ranking_metric in ['routing_weighted_norm', 'fusion']:
                        if num_tokens > 0:
                            # fluc_out *= (num_tokens - 1) / (num_tokens + token_size - 1)
                            # fluc_out += torch.sum((out - baseline_out.unsqueeze(0)).float().pow(2), dim=0) / (num_tokens + token_size)
                            fluc_out *= (num_tokens ) / (num_tokens + token_size)
                            fluc_out += torch.sum(torch.norm(out[route_indx==1]*output_routing, p=2, dim=1)) / (num_tokens + token_size)
                    elif args.expert_ranking_metric in ['io_fluctuation',]:
                        if num_tokens > 0:
                            fluc_out *= (num_tokens - 1) / (num_tokens + token_size - 1)
                            inp = _input[0]
                            inp = inp.view(-1, inp.shape[-1])
                            fluc_out += torch.sum((inp - out).float().pow(2), dim=0) / (num_tokens + token_size)
                    elif args.expert_ranking_metric in ['token_fluctuation', 'intermediate_fluctuation', 'rs_intermediate']:
                        baseline_inp *= num_tokens / (num_tokens + token_size)
                        baseline_inp += torch.sum(inp, dim=0) / (num_tokens + token_size)
                        if num_tokens > 0:
                            fluc_inp *= (num_tokens - 1) / (num_tokens + token_size - 1)
                            fluc_inp += torch.sum((inp - baseline_inp.unsqueeze(0))**2, dim=0) / (num_tokens + token_size)
                    
                # write back stats
                bias_stats[expert_name]["num_tokens"] += token_size
                bias_stats[expert_name]["baseline_out"] = baseline_out
                if args.expert_ranking_metric in ['routing_weighted_norm', 'io_fluctuation', 'fusion']:
                    bias_stats[expert_name]['routing_weighted_norm'] = fluc_out
                elif args.expert_ranking_metric in ['token_fluctuation', 'intermediate_fluctuation', 'rs_intermediate']:
                    bias_stats[expert_name]["baseline_inp"] = baseline_inp
                    bias_stats[expert_name]["input_feature_norm"] = fluc_inp
        
        return stateful_expert_hook

    if args.expert_ranking_metric in ['routing_score', 'fusion', 'rs_intermediate']:
        routing_stats = {}
        def create_gate_hook(layer_idx):
            def stateful_gate_hook(module, _input, _output):
                batch_size = _input[0].shape[0]
                if 'olmoe' in model.config.model_type:
                    router_logits = _output
                    topk_weight = F.softmax(router_logits, dim=1, dtype=torch.float)
                    topk_weight, topk_idx = torch.topk(topk_weight, model.config.num_experts_per_tok, dim=-1)
                    
                    routing_stats[layer_idx]["topk_idx"] = topk_idx
                    routing_stats[layer_idx]["topk_weight"] = topk_weight
                    topk_idx, topk_weight = topk_idx[layer_attn_mask[layer_idx]],topk_weight[layer_attn_mask[layer_idx]]
               

                assert topk_idx.dim() == 2
                # token_size = topk_idx.shape[0]

                routing_weights = torch.zeros(
                    (topk_weight.shape[0], num_experts),
                    device=topk_weight.device, dtype=torch.float
                )
                # num_tokens_per_expert = torch.zeros_like(routing_weights)

                routing_weights = torch.scatter(routing_weights, dim=1, index=topk_idx, src=torch.ones_like(topk_weight).to(torch.float))
                # num_tokens_per_expert = torch.scatter_add(num_tokens_per_expert, dim=1, index=topk_idx, src=torch.ones_like(topk_weight))
                # num_tokens_per_expert = torch.sum(num_tokens_per_expert, dim=0)

                scores = routing_stats[layer_idx]["scores"]
                num_tokens = routing_stats[layer_idx]["num_tokens"]
                
                # scores *= num_tokens / (num_tokens + num_tokens_per_expert)
                # scores += torch.sum(routing_weights, dim=0) / (num_tokens + num_tokens_per_expert)
                
                scores *= num_tokens / (num_tokens + batch_size)
                #scores += torch.sum(routing_weights, dim=0) / (num_tokens + batch_size)
                scores += torch.sum(routing_weights, dim=0) / (num_tokens + batch_size)
                routing_stats[layer_idx]["num_tokens"] += batch_size #num_tokens_per_expert
                routing_stats[layer_idx]["scores"] = scores
                
            return stateful_gate_hook
    
        
        def create_attn_hook(layer_idx):
            def stateful_attn_hook(module, _input, _output):
                #batch_size = _input[0].shape[0]
                
                if 'olmoe' in model.config.model_type:
                    attn_output, attn_weights = _output[:2]
                    delete_num = int(attn_weights.shape[3]*(1-args.tau))
                    attn_self_attention = attn_weights.max(dim=1).values.squeeze()
                    attn_self_attention = attn_self_attention[-1]
                    topk = torch.sort(attn_self_attention * attn_self_attention, dim=-1).indices[:delete_num]
                    topk_idx = torch.ones(attn_weights.shape[3], dtype=torch.bool, device=attn_weights.device)
                    topk_idx[topk] = False
                    layer_attn_mask[layer_idx]=topk_idx
                
            return stateful_attn_hook

        
    for i in valid_moe_layer_indices:
        layer = model.model.layers[i]
        mlp = model.model.layers[i].mlp
        if args.expert_ranking_metric in ['routing_score', 'fusion','rs_intermediate']:
            routing_stats[i] = {
                "scores": torch.zeros(num_experts, device=next(mlp.parameters()).device, dtype=torch.float),
                "num_tokens": torch.zeros(num_experts, device=next(mlp.parameters()).device, dtype=torch.float),
                "topk_idx": None,
                "topk_weight": None
            }
            if "ling" in model.config.model_type:
                handle = layer.attention.register_forward_hook(create_attn_hook(i))
            else:
                handle = layer.self_attn.register_forward_hook(create_attn_hook(i))
               
            handles.append(handle)

            handle = mlp.gate.register_forward_hook(create_gate_hook(i))
            handles.append(handle)
            
        for e_idx in range(len(mlp.experts)):
            expert_name = f"layers.{i}.experts.{e_idx}"
            bias_stats[expert_name] = {
                "num_tokens": 0,
                "baseline_out": torch.zeros(hidden_size, device=next(mlp.parameters()).device, dtype=torch.float),
            }

            if args.expert_ranking_metric in ['routing_weighted_norm', 'io_fluctuation', 'fusion']:
                bias_stats[expert_name]['routing_weighted_norm'] = torch.zeros(hidden_size, device=next(mlp.parameters()).device, dtype=torch.float)
                handle = mlp.experts[e_idx].register_forward_hook(create_expert_hook(expert_name))
                handles.append(handle)
            elif args.expert_ranking_metric in ['token_fluctuation']:
                bias_stats[expert_name]["baseline_inp"] = torch.zeros(hidden_size, device=next(mlp.parameters()).device, dtype=torch.float)
                bias_stats[expert_name]["input_feature_norm"] = torch.zeros(hidden_size, device=next(mlp.parameters()).device, dtype=torch.float)
                handle = mlp.experts[e_idx].register_forward_hook(create_expert_hook(expert_name))
                handles.append(handle)
            elif args.expert_ranking_metric in ['intermediate_fluctuation']:
                bias_stats[expert_name] = {
                    "num_tokens": 0,
                    "baseline_out": torch.zeros(hidden_size, device=next(mlp.parameters()).device, dtype=torch.float),
                    "baseline_inp": torch.zeros(intermediate_size, device=next(mlp.parameters()).device, dtype=torch.float),
                    "input_feature_norm": torch.zeros(intermediate_size, device=next(mlp.parameters()).device, dtype=torch.float),
                }
                handle = mlp.experts[e_idx].down_proj.register_forward_hook(create_expert_hook(expert_name))
                handles.append(handle)

    data_iter = iter(train_dataloader)
    for step in tqdm(range(args.max_steps), desc="collecting accumulated stats"):
        batch = next(data_iter)
        #fold = args.max_steps//128
        with torch.no_grad():
            #for i in range(fold):
                batch = {k: v.to("cuda:0") for k, v in batch.items()}
                model(**batch)
    
    for handle in handles:
        handle.remove()
    
    #########################
    # Collect pruning metrics
    metric_list = {}
    
    if args.expert_ranking_metric == 'routing_score':
        for layer_idx in valid_moe_layer_indices:
            metric_list[layer_idx] = routing_stats[layer_idx]["scores"]
    elif args.expert_ranking_metric in ['routing_weighted_norm', 'io_fluctuation']:
        for layer_idx in valid_moe_layer_indices:
            fluc_list = [bias_stats[f'layers.{layer_idx}.experts.{e_idx}']['routing_weighted_norm'] for e_idx in range(num_experts)]
            output_fluc = torch.stack(fluc_list)
            metric_list[layer_idx] = torch.norm(output_fluc, dim=1)
    elif args.expert_ranking_metric == 'fusion':
        for layer_idx in valid_moe_layer_indices:
            fluc_list = [bias_stats[f'layers.{layer_idx}.experts.{e_idx}']['routing_weighted_norm'] for e_idx in range(num_experts)]
            # num_experts
            output_fluc = torch.norm(torch.sqrt(torch.stack(fluc_list)), dim=1)
            metric_list[layer_idx] = routing_stats[layer_idx]["scores"]
    elif args.expert_ranking_metric=='token_fluctuation':
        for layer_idx in valid_moe_layer_indices:
            fluc_list = [bias_stats[f'layers.{layer_idx}.experts.{e_idx}']['input_feature_norm'] for e_idx in range(num_experts)]
            input_fluc = torch.stack(fluc_list)
            metric_list[layer_idx] = torch.norm(input_fluc, dim=1)
    elif args.expert_ranking_metric=='intermediate_fluctuation':
        for layer_idx in valid_moe_layer_indices:
            scores = []
            for e_idx in range(num_experts):
                inp_fluc = bias_stats[f'layers.{layer_idx}.experts.{e_idx}']['input_feature_norm']
                mlp = model.model.layers[layer_idx].mlp
                intermediate_score = inp_fluc * torch.sum(mlp.experts[e_idx].down_proj.weight.data.pow(2), dim=0)
                scores.append(intermediate_score)
            metric_list[layer_idx] = torch.norm(torch.stack(scores), dim=1)
    else:
        raise ValueError(f"unknow ranking metric: {args.expert_ranking_metric}")

    ###########################
    # Collect pruning threshold
    expert_sort=[]
    if args.expert_ranking_scope == 'model':
        metrics = [metric.to('cuda:0') for metric in metric_list.values()]
        metric = torch.cat(metrics)
        sorted_scores, _ = torch.sort(metric, descending=True)
        threshold_val = sorted_scores[math.ceil(len(sorted_scores)*args.preserve_n_experts/num_experts)]
        threshold = {layer_idx: threshold_val for layer_idx in valid_moe_layer_indices}
    else:
        threshold = {}
        for layer_idx in metric_list:
            metric = metric_list[layer_idx]
            sorted_scores, indices = torch.sort(metric, descending=True)
            expert_sort.append(indices.unsqueeze(0))
            threshold[layer_idx] = sorted_scores[args.preserve_n_experts]

    #################
    # Run pruning process
    approximate_experts = {}
    approximate_expert_init_tokens = {}
    for layer_idx in valid_moe_layer_indices:
        layer_metric = (metric_list[layer_idx]).to(threshold[layer_idx].device)
        expert_mask = layer_metric > threshold[layer_idx]

        expert_indicator_list = expert_mask.tolist()
        approximate_experts[layer_idx] = []
        approximate_expert_init_tokens[layer_idx] = []
        for expert_idx, is_preserved in enumerate(expert_indicator_list):
            if not is_preserved:
                approximate_experts[layer_idx].append(expert_idx)
                approximate_expert_init_tokens[layer_idx].append(
                    bias_stats[expert_name]['num_tokens'] if args.enable_novice_evolving else 0
                )
                expert_name = f"layers.{layer_idx}.experts.{expert_idx}"
                novice = novice_cls(model.config, is_approx=True, 
                    acc_tokens=bias_stats[expert_name]['num_tokens'] if args.enable_novice_evolving else 0)
                novice.requires_grad_(False)
                novice.approx_value.copy_(
                    torch.zeros_like(bias_stats[expert_name]["baseline_out"]) 
                )                
                model.model.layers[layer_idx].mlp.experts[expert_idx] = novice.bfloat16().to(next(model.model.layers[layer_idx].mlp.parameters()).device)
                # model.model.layers[layer_idx].mlp.experts[expert_idx].gate_proj.weight.data=torch.zeros_like(mlp.experts[expert_idx].gate_proj.weight.data)
                # model.model.layers[layer_idx].mlp.experts[expert_idx].up_proj.weight.data=torch.zeros_like(mlp.experts[expert_idx].up_proj.weight.data)
                # model.model.layers[layer_idx].mlp.experts[expert_idx].down_proj.weight.data=torch.zeros_like(mlp.experts[expert_idx].down_proj.weight.data)
    # update configs
    model.config.approximate_experts = approximate_experts
    model.config.approximate_expert_init_tokens = approximate_expert_init_tokens
    
    # model.cpu()
    torch.cuda.empty_cache()
    return model


@torch.no_grad()
def expert_prune_by_step(args, model, train_dataloader):
    # Move the model to the GPU 
    #model.cuda()
    # Retrieves the index of the currently active GPU device
    device = torch.cuda.current_device()

    handles = []
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    # if "deepseek_v3" in model.config.model_type:
    #     mlp_module_class = DeepseekV3MLP
    #     num_experts = model.config.n_routed_experts
    #     intermediate_size = model.config.moe_intermediate_size
    # elif "deepseek_v2" in model.config.model_type:
    #     mlp_module_class = DeepseekV2MLP
    #     num_experts = model.config.n_routed_experts
    #     intermediate_size = model.config.moe_intermediate_size
    if "olmoe" in model.config.model_type:
        mlp_module_class = OlmoeMLP
        num_experts = model.config.num_experts
        intermediate_size = model.config.intermediate_size
    # elif "deepseek" in model.config.model_type:
    #     mlp_module_class = DeepseekMLP
    #     num_experts = model.config.n_routed_experts
    #     intermediate_size = model.config.moe_intermediate_size
    # elif "qwen" in model.config.model_type:
    #     mlp_module_class = Qwen3MoeMLP
    #     num_experts = model.config.num_experts
    #     intermediate_size = model.config.moe_intermediate_size
    # elif "glm" in model.config.model_type:
    #     mlp_module_class = Glm4MoeMLP
    #     num_experts = model.config.n_routed_experts
    #     intermediate_size = model.config.moe_intermediate_size
    # elif "ling" in model.config.model_type:
    #     mlp_module_class = BailingMoeV2MLP
    #     num_experts = model.config.num_experts
    #     intermediate_size = model.config.moe_intermediate_size
    else:
        raise ValueError(f"unknow model type: {model.config.model_type}")

    # Identify MoE layer
    if "deepseek" in model.config.model_type:
        valid_moe_layer_indices = [
            layer_idx for layer_idx in range(num_layers) 
            if (
                model.config.n_routed_experts is not None and
                layer_idx >= model.config.first_k_dense_replace and 
                layer_idx % model.config.moe_layer_freq == 0
            )
        ]
    elif "olmoe" in model.config.model_type:
        valid_moe_layer_indices = list(range(num_layers))
    elif "qwen" in model.config.model_type:
        valid_moe_layer_indices = [layer_idx for layer_idx in range(num_layers) if (layer_idx not in model.config.mlp_only_layers) and (
            model.config.num_experts > 0 and (layer_idx + 1) % model.config.decoder_sparse_step == 0
        )]
    elif "glm" in model.config.model_type:
        valid_moe_layer_indices = [
            layer_idx for layer_idx in range(num_layers) 
            if (
                model.config.n_routed_experts is not None and
                layer_idx >= model.config.first_k_dense_replace 
            )
        ]
    elif "ling" in model.config.model_type:
        valid_moe_layer_indices = [
            layer_idx for layer_idx in range(num_layers) 
            if (model.config.num_experts is not None and layer_idx >= model.config.first_k_dense_replace)
        ]
    #########################################
    # Create hooks to collect pruning metrics
    def merge(target_indices, reference_indices):
        # Create zero tensor with same length as target_indices
        result = torch.zeros_like(target_indices)

        # Find intersection positions between target_indices and reference_indices (using broadcast and element-wise comparison)
        mask = (target_indices.unsqueeze(1) == reference_indices).any(dim=1)  # shape: [m]

        # Set intersection positions to 1, keep others as 0
        result[mask] = 1

        return result

    bias_stats = {}
    layer_attn_mask = {}
    def create_expert_hook(expert_name):
        def stateful_expert_hook(module, _input, _output):
            expert_output = _output
            expert_output = expert_output.view(-1, expert_output.shape[-1])
            expert_input = _input[0]
            expert_input = expert_input.view(-1, expert_input.shape[-1])
            token_count = expert_output.shape[0]
            layer_idx, expert_idx = int(expert_name.split('.')[1]), int(expert_name.split('.')[-1])
            temp_routing_state = routing_stats[layer_idx]["topk_idx"].clone()
            temp_routing_state[~layer_attn_mask[layer_idx]] = -1
            if not torch.all(temp_routing_state == expert_idx):
                routing_indices = (routing_stats[layer_idx]["topk_idx"] == expert_idx).view(-1)
                temp_routing_indices = torch.where((temp_routing_state == expert_idx).view(-1))[0] // model.config.num_experts_per_tok
                temp_routing_indices_all = torch.where(routing_indices)[0] // model.config.num_experts_per_tok
                route_mask = merge(temp_routing_indices_all, temp_routing_indices)
                output_routing = routing_stats[layer_idx]["topk_weight"].view(-1)[torch.where((temp_routing_state == expert_idx).view(-1))[0]].unsqueeze(dim=-1)
                # retrieve stats
                expert_token_count = bias_stats[expert_name]["num_tokens"]
                baseline_out = bias_stats[expert_name]["baseline_out"]
                
                routing_weighted_norm = bias_stats[expert_name]["routing_weighted_norm"]
                if expert_token_count > 0:
                    # update moving average and fluctuation
                    baseline_out *= expert_token_count / (expert_token_count + token_count)
                    baseline_out += torch.sum(expert_output.float(), dim=0) / (expert_token_count + token_count)
                    
                    if expert_token_count > 0:
                        routing_weighted_norm *= (expert_token_count) / (expert_token_count + token_count)
                        routing_weighted_norm += torch.sum(torch.norm(expert_output[route_mask == 1] * output_routing, p=2, dim=1)) / (expert_token_count + token_count)

                # write back stats
                bias_stats[expert_name]["num_tokens"] += token_count
                bias_stats[expert_name]["baseline_out"] = baseline_out
                bias_stats[expert_name]['routing_weighted_norm'] = routing_weighted_norm
                
        return stateful_expert_hook

    if args.expert_ranking_metric in ['routing_score', 'fusion', 'rs_intermediate']:
        routing_stats = {}
        def create_gate_hook(layer_idx):
            def stateful_gate_hook(module, _input, _output):
                batch_size = _input[0].shape[0]
                if 'olmoe' in model.config.model_type:
                    router_logits = _output
                    topk_weight = F.softmax(router_logits, dim=1, dtype=torch.float)
                    topk_weight, topk_idx = torch.topk(topk_weight, model.config.num_experts_per_tok, dim=-1)
                    
                    routing_stats[layer_idx]["topk_idx"] = topk_idx
                    routing_stats[layer_idx]["topk_weight"] = topk_weight
                    topk_idx, topk_weight = topk_idx[layer_attn_mask[layer_idx]],topk_weight[layer_attn_mask[layer_idx]]
                
                assert topk_idx.dim() == 2
                # token_size = topk_idx.shape[0]

                routing_weights = torch.zeros(
                    (topk_weight.shape[0], num_experts),
                    device=topk_weight.device, dtype=torch.float
                )
                # num_tokens_per_expert = torch.zeros_like(routing_weights)

                routing_weights = torch.scatter(routing_weights, dim=1, index=topk_idx, src=torch.ones_like(topk_weight).to(torch.float))
                # num_tokens_per_expert = torch.scatter_add(num_tokens_per_expert, dim=1, index=topk_idx, src=torch.ones_like(topk_weight))
                # num_tokens_per_expert = torch.sum(num_tokens_per_expert, dim=0)

                scores = routing_stats[layer_idx]["scores"]
                num_tokens = routing_stats[layer_idx]["num_tokens"]
                
                # scores *= num_tokens / (num_tokens + num_tokens_per_expert)
                # scores += torch.sum(routing_weights, dim=0) / (num_tokens + num_tokens_per_expert)
                
                scores *= num_tokens / (num_tokens + batch_size)
                #scores += torch.sum(routing_weights, dim=0) / (num_tokens + batch_size)
                scores += torch.sum(routing_weights, dim=0) / (num_tokens + batch_size)
                routing_stats[layer_idx]["num_tokens"] += batch_size #num_tokens_per_expert
                routing_stats[layer_idx]["scores"] = scores
                
            return stateful_gate_hook
    
        
        def create_attn_hook(layer_idx):
            def stateful_attn_hook(module, _input, _output):
                #batch_size = _input[0].shape[0]
                if 'olmoe' in model.config.model_type:
                    attn_output, attn_weights = _output[:2]
                    delete_num = int(attn_weights.shape[3]*(1-args.tau))
                    attn_self_attention = attn_weights.max(dim=1).values.squeeze()
                    attn_self_attention = attn_self_attention[-1]
                    topk = torch.sort(attn_self_attention * attn_self_attention, dim=-1).indices[:delete_num]
                    topk_idx = torch.ones(attn_weights.shape[3], dtype=torch.bool, device=attn_weights.device)
                    topk_idx[topk] = False
                    layer_attn_mask[layer_idx]=topk_idx
                
            return stateful_attn_hook

        
    for i in valid_moe_layer_indices:
        layer = model.model.layers[i]
        mlp = model.model.layers[i].mlp
        if args.expert_ranking_metric in ['routing_score', 'fusion','rs_intermediate']:
            routing_stats[i] = {
                "scores": torch.zeros(num_experts, device=next(mlp.parameters()).device, dtype=torch.float),
                "num_tokens": torch.zeros(num_experts, device=next(mlp.parameters()).device, dtype=torch.float),
                "topk_idx": None,
                "topk_weight": None
            }
            if 'ling' in model.config.model_type:
                handle = layer.attention.register_forward_hook(create_attn_hook(i))
            else:
                handle = layer.self_attn.register_forward_hook(create_attn_hook(i))
            handles.append(handle)

            handle = mlp.gate.register_forward_hook(create_gate_hook(i))
            handles.append(handle)
            
        for e_idx in range(len(mlp.experts)):
            expert_name = f"layers.{i}.experts.{e_idx}"
            bias_stats[expert_name] = {
                "num_tokens": 0,
                "baseline_out": torch.zeros(hidden_size, device=next(mlp.parameters()).device, dtype=torch.float),
            }

            if args.expert_ranking_metric in ['routing_weighted_norm', 'io_fluctuation', 'fusion']:
                bias_stats[expert_name]['routing_weighted_norm'] = torch.zeros(hidden_size, device=next(mlp.parameters()).device, dtype=torch.float)
                handle = mlp.experts[e_idx].register_forward_hook(create_expert_hook(expert_name))
                handles.append(handle)
            elif args.expert_ranking_metric in ['token_fluctuation']:
                bias_stats[expert_name]["baseline_inp"] = torch.zeros(hidden_size, device=next(mlp.parameters()).device, dtype=torch.float)
                bias_stats[expert_name]["input_feature_norm"] = torch.zeros(hidden_size, device=next(mlp.parameters()).device, dtype=torch.float)
                handle = mlp.experts[e_idx].register_forward_hook(create_expert_hook(expert_name))
                handles.append(handle)
            elif args.expert_ranking_metric in ['intermediate_fluctuation']:
                bias_stats[expert_name] = {
                    "num_tokens": 0,
                    "baseline_out": torch.zeros(hidden_size, device=next(mlp.parameters()).device, dtype=torch.float),
                    "baseline_inp": torch.zeros(intermediate_size, device=next(mlp.parameters()).device, dtype=torch.float),
                    "input_feature_norm": torch.zeros(intermediate_size, device=next(mlp.parameters()).device, dtype=torch.float),
                }
                handle = mlp.experts[e_idx].down_proj.register_forward_hook(create_expert_hook(expert_name))
                handles.append(handle)

    data_iter = iter(train_dataloader)
    for step in tqdm(range(args.max_steps), desc="collecting accumulated stats"):
        batch = next(data_iter)
        with torch.no_grad():
            batch = {k: v.to("cuda:0") for k, v in batch.items()}
            model(**batch)
    
    for handle in handles:
        handle.remove()
    
    #########################
    # Collect pruning metrics
    metric_list = {}
    if args.expert_ranking_metric == 'fusion':
        for layer_idx in valid_moe_layer_indices:
            fluc_list = [bias_stats[f'layers.{layer_idx}.experts.{e_idx}']['routing_weighted_norm'] for e_idx in range(num_experts)]
            # num_experts
            output_fluc = torch.norm(torch.sqrt(torch.stack(fluc_list)), dim=1)
            metric_list[layer_idx] = (args.fusion_io_weight * output_fluc) * ((1-args.fusion_io_weight) * routing_stats[layer_idx]["scores"])
    elif args.expert_ranking_metric=='token_fluctuation':
        for layer_idx in valid_moe_layer_indices:
            fluc_list = [bias_stats[f'layers.{layer_idx}.experts.{e_idx}']['input_feature_norm'] for e_idx in range(num_experts)]
            input_fluc = torch.stack(fluc_list)
            metric_list[layer_idx] = torch.norm(input_fluc, dim=1)
    elif args.expert_ranking_metric=='intermediate_fluctuation':
        for layer_idx in valid_moe_layer_indices:
            scores = []
            for e_idx in range(num_experts):
                inp_fluc = bias_stats[f'layers.{layer_idx}.experts.{e_idx}']['input_feature_norm']
                mlp = model.model.layers[layer_idx].mlp
                intermediate_score = inp_fluc * torch.sum(mlp.experts[e_idx].down_proj.weight.data.pow(2), dim=0)
                scores.append(intermediate_score)
            metric_list[layer_idx] = torch.norm(torch.stack(scores), dim=1)
    else:
        raise ValueError(f"unknow ranking metric: {args.expert_ranking_metric}")

    ###########################
    # Collect pruning threshold
    expert_sort=[]
    if args.expert_ranking_scope == 'model':
        metrics = [metric.to('cuda:0') for metric in metric_list.values()]
        metric = torch.cat(metrics)
        sorted_scores, _ = torch.sort(metric, descending=True)
        threshold_val = sorted_scores[math.ceil(len(sorted_scores)*args.preserve_n_experts/num_experts)]
        threshold = {layer_idx: threshold_val for layer_idx in valid_moe_layer_indices}
    else:
        threshold = {}
        for layer_idx in metric_list:
            metric = metric_list[layer_idx]
            sorted_scores, indices = torch.sort(metric, descending=True)
            expert_sort.append(indices.unsqueeze(0))
            threshold[layer_idx] = sorted_scores[args.preserve_n_experts]

    #################
    # Run pruning process
    approximate_experts = {}
    approximate_expert_init_tokens = {}
    out_mask= []
    for layer_idx in valid_moe_layer_indices:
        layer_metric = (metric_list[layer_idx]).to(threshold[layer_idx].device)
        expert_mask = layer_metric > threshold[layer_idx]

        expert_indicator_list = expert_mask.tolist()
        approximate_experts[layer_idx] = []
        approximate_expert_init_tokens[layer_idx] = []
        out_mask.append(expert_indicator_list)
        for expert_idx, is_preserved in enumerate(expert_indicator_list):
            if not is_preserved:
                
                approximate_experts[layer_idx].append(expert_idx)
                approximate_expert_init_tokens[layer_idx].append(
                     0
                )
                expert_name = f"layers.{layer_idx}.experts.{expert_idx}"
                mlp_module = mlp_module_class(model.config, is_approx=True, 
                    acc_tokens=0)
                mlp_module.approx_value.copy_(
                    torch.zeros_like(bias_stats[expert_name]["baseline_out"]) if args.no_bias else bias_stats[expert_name]["baseline_out"]
                )                
                model.model.layers[layer_idx].mlp.experts[expert_idx] = mlp_module.bfloat16().to(next(model.model.layers[layer_idx].mlp.parameters()).device)

    # update configs
    model.config.approximate_experts = approximate_experts
    model.config.approximate_expert_init_tokens = approximate_expert_init_tokens
    
    # model.cpu()
    torch.cuda.empty_cache()
    return model

def finetune(args, model, train_dataloader):
    from torch.cuda.amp import autocast
    data_iter = iter(train_dataloader)
    #device = torch.cuda.current_device()
    # ===== 1. Enable BF16 mixed precision =====

    # model = model.to(torch.bfloat16)  # Convert base model to BF16
    # model.to('cuda')
    # ===== 2. Freeze non-gate parameters =====
    for name, param in model.named_parameters():
        # if not ('gate' in name and 'gate_proj' not in name):
        #     param.requires_grad = False
        # else:
        #     param.data = param.data.to(torch.bfloat16)
        if 'approx_value' in name or ('gate' in name and 'gate_proj' not in name):
            pass
        else:
            param.requires_grad = False
            # param.data = param.data.to(torch.bfloat16)  # Ensure gate parameters are BF16

    # ===== 3. BF16 optimizer configuration =====
    gate_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        gate_params,
        lr=1e-4,  # Suggest slightly larger (1e-3 ~ 5e-4)
        weight_decay=0.1,
        eps=1e-6  # BF16 requires larger epsilon to prevent numerical instability
    )
    
    # ===== 4. BF16 training loop =====
    model.train()
    data_iter = iter(train_dataloader)
    
    for _ in tqdm(range(args.max_steps), desc="BF16 Gate Finetuning"):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            batch = next(data_iter)
        
        # Convert input data to BF16 (note: labels remain int/long)
        batch = {k: v.to("cuda:0") for k, v in batch.items()}
        
        # Mixed precision training context
        with autocast(dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss
        
        # BF16 optimization step (gradient scaling + unscaling)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
     
def expert_prune(args, model, train_dataloader,tokenizer):    

    if args.expert_prune_metric == 'routing_score':
        expert_prune_by_routing_score(args, model, train_dataloader)
    elif args.expert_prune_metric == 'mc_smoe':
        expert_prune_by_mc_smoe(args, model, train_dataloader)
    elif args.expert_prune_metric == 'mone':
        expert_prune_by_mone(args, model, train_dataloader)
    elif args.expert_prune_metric == 'step':
        expert_prune_by_step(args, model, train_dataloader)
    elif args.expert_prune_metric == 'freq':
        expert_prune_by_freq(args, model, train_dataloader)
    # elif args.expert_prune_metric == 'step_ll':
    #     expert_prune_by_step_ll(args, model, train_dataloader,tokenizer)
    # elif args.expert_prune_metric == 'step_sc':
    #     expert_prune_by_step_sc(args, model, train_dataloader,tokenizer)
    # elif args.expert_prune_metric == 'freq_ll':
    #     expert_prune_by_freq_ll(args, model, train_dataloader,tokenizer)
    # elif args.expert_prune_metric == 'freq_sc':
    #     expert_prune_by_freq_sc(args, model, train_dataloader,tokenizer)
    # elif args.expert_prune_metric == 'step_norm':
    #     expert_prune_by_step_norm(args, model, train_dataloader)
    # elif args.expert_prune_metric == 'step_change':
    #     expert_prune_by_step_change(args, model, train_dataloader)
    # elif args.expert_prune_metric == 'step_grad':
    #     expert_prune_by_step_grad(args, model, train_dataloader)
    
    if 'step' in args.expert_prune_metric:   
        finetune(args, model, train_dataloader)
    
    