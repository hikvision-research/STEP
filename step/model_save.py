import argparse

import torch
import torch.nn.functional

from step.utils import prepare_model_and_tokenizer
from step.pruning.expert_prune import expert_prune




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="deepseek-ai/DeepSeek-V2-Lite-Chat",)

    # parser.add_argument("--mod_type", type=str, default=None, choices=['staged', 'integrated'])
    # parser.add_argument("--staged_mod_topk", type=int, default=2048)

    # build a dataset for pruning
    parser.add_argument("--dataset_name_or_path", type=str, default="HuggingFaceFW/fineweb",) # "allenai/OLMoE-mix-0924"
    parser.add_argument("--dataset_config_name", type=str, default=None) # None
    parser.add_argument("--data_type", type=str, default=None)
    parser.add_argument("--streaming_dataset", action='store_true')

    parser.add_argument("--block_size", type=int, default=4*1024,)
    parser.add_argument("--max_steps", type=int, default=100,)

    
    # MoE expert pruning related arguments
    parser.add_argument("--expert_prune", action="store_true",)
    parser.add_argument("--preserve_n_experts", type=int, default=30, help="Number of experts to preserve")
    parser.add_argument("--expert_prune_metric", type=str, default='routing_score', choices=['routing_score', 'mc_smoe', 'mone'])
    parser.add_argument("--expert_ranking_scope", type=str, default="model", choices=['model', 'layer'])
    parser.add_argument("--expert_ranking_metric", type=str, default="routing_score", choices=['routing_score', 'output_fluctuation', 'io_fluctuation', 'fusion', 'token_fluctuation', 'intermediate_fluctuation'])
    parser.add_argument("--enable_novice_evolving", action='store_true')
    parser.add_argument("--fusion_io_weight", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=1.0, help='Token retention ratio')
    parser.add_argument("--no_bias", action='store_true')
    
    parser.add_argument("--compressed_model_save_path", type=str, default=" allenai/OLMoE-1B-7B-0125-pruned",)
    return parser.parse_args()

from minipile import get_calib_dataloder
def get_dataloader(args, tokenizer):
    return get_calib_dataloder(
        dataset="c4",
        tokenizer=tokenizer,
        max_block_size=2048,
        n_blocks_for_stat=128, # 32, 128
        batch_size=1,
        num_workers=4,
    )

def main():
    args = parse_args()

    model, tokenizer = prepare_model_and_tokenizer(args.model_name_or_path, mode='prune')
    model.eval()

    train_dataloader = None
    train_dataloader = get_dataloader(args,tokenizer)

    
    expert_prune(args, model, train_dataloader)
    new_model = model
    if hasattr(new_model.config, "auto_map"):
        del new_model.config.auto_map
    if "OLMoE" in args.model_name_or_path:
        new_model.config.auto_map = {
    "AutoConfig": "configuration_olmoe.OlmoeConfig",
    "AutoModel": "modeling_olmoe.OlmoeModel",
    "AutoModelForCausalLM": "modeling_olmoe.OlmoeForCausalLM"
    }     
           
    # Save
    new_model.save_pretrained(args.compressed_model_save_path)
    tokenizer.save_pretrained(args.compressed_model_save_path)


if __name__ == "__main__":
    # model.cuda()
    # text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
    # inputs = tokenizer(text, return_tensors="pt")
    # outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)
    # result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(result)
    # exit(0)
    main()
    