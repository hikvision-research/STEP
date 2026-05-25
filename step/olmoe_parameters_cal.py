import torch
import torch.nn as nn


def calculate_olmoe_parameters(config):
    """
    Calculate total and active parameters of OLMoE model.

    Total parameters: all parameters in the model.
    Active parameters: parameters actually used during inference (sparse MoE).
    """
    total_params = 0
    active_params = 0

    # 1. Embedding layer parameters
    vocab_size = config["vocab_size"]
    hidden_size = config["hidden_size"]
    embedding_params = vocab_size * hidden_size
    total_params += embedding_params
    active_params += embedding_params  # Embedding layer is always active

    # 2. Attention layer parameters (per layer)
    num_layers = config["num_hidden_layers"]
    num_heads = config["num_attention_heads"]
    head_dim = hidden_size // num_heads
    num_kv_heads = config["num_key_value_heads"]

    # QKV projection parameters
    q_proj_params = hidden_size * (num_heads * head_dim)
    k_proj_params = hidden_size * (num_kv_heads * head_dim)
    v_proj_params = hidden_size * (num_kv_heads * head_dim)
    o_proj_params = (num_heads * head_dim) * hidden_size

    # RMSNorm parameters (Q and K normalization)
    q_norm_params = hidden_size  # Q normalization
    k_norm_params = num_kv_heads * head_dim  # K normalization

    # Attention parameters per layer
    attention_params_per_layer = q_proj_params + k_proj_params + v_proj_params + o_proj_params + q_norm_params + k_norm_params

    # 3. MLP expert parameters (per layer)
    intermediate_size = config["intermediate_size"]
    num_experts = config["num_experts"]

    # Parameters per expert
    gate_proj_params = hidden_size * intermediate_size
    up_proj_params = hidden_size * intermediate_size
    down_proj_params = intermediate_size * hidden_size

    mlp_params_per_expert = gate_proj_params + up_proj_params + down_proj_params
    mlp_params_per_layer = num_experts * mlp_params_per_expert

    # 4. Router gate parameters (per layer)
    router_params_per_layer = hidden_size * num_experts

    # 5. Layer normalization parameters (per layer)
    # Input layer normalization
    input_norm_params = hidden_size
    # Post-attention layer normalization
    post_attention_norm_params = hidden_size

    norm_params_per_layer = input_norm_params + post_attention_norm_params

    # 6. Final layer normalization parameters
    final_norm_params = hidden_size

    # 7. Language model head parameters
    lm_head_params = hidden_size * vocab_size

    # Calculate total parameters
    # Attention layers (all layers)
    total_attention_params = num_layers * attention_params_per_layer

    # MLP layers (all layers) - all experts
    total_mlp_params = num_layers * mlp_params_per_layer

    # Router gates (all layers)
    total_router_params = num_layers * router_params_per_layer

    # Layer normalization (all layers)
    total_norm_params = num_layers * norm_params_per_layer + final_norm_params

    # Total parameters
    total_params = (embedding_params + total_attention_params + total_mlp_params +
                    total_router_params + total_norm_params + lm_head_params)

    # Calculate actually active parameters (MoE activates only some experts)
    num_experts_per_tok = config["num_experts_per_tok"]

    # Active MLP parameter ratio
    active_mlp_ratio = num_experts_per_tok / num_experts

    # Active MLP parameters
    active_mlp_params = num_layers * (mlp_params_per_layer * active_mlp_ratio)

    # Router gate parameters are always active
    active_router_params = total_router_params

    # Total active parameters
    active_params = (embedding_params + total_attention_params + active_mlp_params +
                     active_router_params + total_norm_params + lm_head_params)

    return total_params, active_params


def format_number(num):
    """Format number for display."""
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return f"{num}"


# Model configuration
base_config = {
    "_name_or_path": "step23842-hf",
    "architectures": ["OlmoeForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "clip_qkv": None,
    "eos_token_id": 50279,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 1024,
    "max_position_embeddings": 4096,
    "model_type": "olmoe",
    "norm_topk_prob": False,
    "num_attention_heads": 16,
    "num_experts": 64,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 16,
    "num_key_value_heads": 16,
    "output_router_logits": False,
    "pad_token_id": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": None,
    "rope_theta": 10000.0,
    "router_aux_loss_coef": 0.01,
    "tie_word_embeddings": False,
    "torch_dtype": "float32",
    "transformers_version": "4.48.1",
    "use_cache": True,
    "vocab_size": 50304
}

# Calculate parameters

#total_params, active_params = calculate_olmoe_parameters(base_config)


import itertools


num_hidden_layers = [12, 13, 14, 15, 16]
intermediate_sizes = [512, 640, 768, 896, 1024]
hidden_sizes = [1024, 1280, 1536, 1792, 2048]

combinations = list(itertools.product(num_hidden_layers, intermediate_sizes, hidden_sizes))

for i, combo in enumerate(combinations):
    temp_config = {**base_config}
    temp_config["hidden_size"] = combo[2]
    temp_config["intermediate_size"] = combo[1]
    temp_config["num_hidden_layers"] = combo[0]

    total_params, active_params = calculate_olmoe_parameters(temp_config)
    print(f"================{i}-th comb=============================")
    print(f"Total parameters: {total_params:,} ({format_number(total_params)})")
    print(f"Active parameters: {active_params:,} ({format_number(active_params)})")
#



# Output results (commented out examples)
# print("=== OLMoE Model Parameter Statistics ===")
# print(f"Model Configuration:")
# print(f"- Hidden size: {config['hidden_size']}")
# print(f"- Number of layers: {config['num_hidden_layers']}")
# print(f"- Attention heads: {config['num_attention_heads']}")
# print(f"- Number of experts: {config['num_experts']}")
# print(f"- Experts per token: {config['num_experts_per_tok']}")
# print(f"- Vocabulary size: {config['vocab_size']}")
# print()

# print(f"Total parameters: {total_params:,} ({format_number(total_params)})")
# print(f"Active parameters: {active_params:,} ({format_number(active_params)})")
# print(f"Active parameter ratio: {active_params / total_params * 100:.2f}%")
# print(f"MoE sparsity: {1 - active_params / total_params:.2%}")

# # Detailed parameter breakdown
# print("\n=== Detailed Parameter Breakdown ===")
# embedding_params = config["vocab_size"] * config["hidden_size"]
# num_layers = config["num_hidden_layers"]

# # Attention parameter calculation
# num_heads = config["num_attention_heads"]
# head_dim = config["hidden_size"] // num_heads
# num_kv_heads = config["num_key_value_heads"]

# q_proj_params = config["hidden_size"] * (num_heads * head_dim)
# k_proj_params = config["hidden_size"] * (num_kv_heads * head_dim)
# v_proj_params = config["hidden_size"] * (num_kv_heads * head_dim)
# o_proj_params = (num_heads * head_dim) * config["hidden_size"]
# attention_params_per_layer = q_proj_params + k_proj_params + v_proj_params + o_proj_params + config["hidden_size"] + (
#             num_kv_heads * head_dim)

# # MLP parameter calculation
# intermediate_size = config["intermediate_size"]
# mlp_params_per_expert = (config["hidden_size"] * intermediate_size) * 2 + (intermediate_size * config["hidden_size"])
# mlp_params_per_layer = config["num_experts"] * mlp_params_per_expert

# print(f"1. Embedding layer: {format_number(embedding_params)}")
# print(f"2. Attention layers ({num_layers} layers): {format_number(num_layers * attention_params_per_layer)}")
# print(f"3. MLP expert layers ({num_layers} layers): {format_number(num_layers * mlp_params_per_layer)}")
# print(f"4. Router gates ({num_layers} layers): {format_number(num_layers * config['hidden_size'] * config['num_experts'])}")
# print(f"5. Layer normalization: {format_number((num_layers * 2 + 1) * config['hidden_size'])}")
# print(f"6. Language model head: {format_number(config['hidden_size'] * config['vocab_size'])}")
