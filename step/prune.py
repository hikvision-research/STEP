"""
Main entry point for MoE expert pruning.

This script provides a command-line interface for pruning Mixture-of-Experts (MoE)
models using various pruning strategies including STEP, frequency-based, and MONE.

Example usage:
    python step/prune.py \
        --model_name_or_path allenai/OLMoE-1B-7B-0125 \
        --expert_prune \
        --preserve_n_experts 32 \
        --expert_prune_metric step \
        --expert_ranking_metric fusion \
        --max_steps 128 \
        --tau 0.5
"""

import argparse

import torch
import torch.nn.functional as F

from step.lib.eval import eval_ppl, eval_zero_shot
from step.minipile import get_calib_dataloder
from step.pruning.expert_prune import expert_prune
from step.utils import prepare_model_and_tokenizer


def parse_args():
    """Parse command-line arguments for MoE pruning.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Prune MoE models by removing less important experts"
    )

    # Model configuration
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="allenai/OLMoE-1B-7B-0125",
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        default="c4",
        help="Calibration dataset name or path (c4, code-alpaca)",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="Dataset configuration name",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default=None,
        help="Type of data format",
    )
    parser.add_argument(
        "--streaming_dataset",
        action="store_true",
        help="Use streaming mode for dataset loading",
    )

    # Data loading configuration
    parser.add_argument(
        "--block_size",
        type=int,
        default=4096,
        help="Block size for tokenization (default: 4*1024)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100,
        help="Number of calibration samples for pruning",
    )

    # Expert pruning configuration
    parser.add_argument(
        "--expert_prune",
        action="store_true",
        help="Enable expert pruning",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save the pruned model",
    )
    parser.add_argument(
        "--preserve_n_experts",
        type=int,
        default=30,
        help="Number of experts to preserve after pruning",
    )
    parser.add_argument(
        "--expert_prune_metric",
        type=str,
        default="routing_score",
        choices=["routing_score", "mc_smoe", "mone", "step", "freq"],
        help="Metric for expert pruning (default: routing_score)",
    )
    parser.add_argument(
        "--expert_ranking_scope",
        type=str,
        default="model",
        choices=["model", "layer"],
        help="Scope for expert ranking: model-level or layer-level",
    )
    parser.add_argument(
        "--expert_ranking_metric",
        type=str,
        default="routing_score",
        choices=[
            "routing_score",
            "output_fluctuation",
            "io_fluctuation",
            "fusion",
            "token_fluctuation",
            "intermediate_fluctuation",
        ],
        help="Metric for ranking experts",
    )
    parser.add_argument(
        "--enable_novice_evolving",
        action="store_true",
        help="Enable novice evolving during pruning",
    )
    parser.add_argument(
        "--fusion_io_weight",
        type=float,
        default=0.5,
        help="Weight for fusion of input/output metrics",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="Token retention ratio for STEP method (0.0-1.0)",
    )
    parser.add_argument(
        "--no_bias",
        action="store_true",
        help="Do not add bias terms after pruning",
    )

    # Output configuration
    parser.add_argument(
        "--compressed_model_save_path",
        type=str,
        default="allenai/OLMoE-1B-7B-0125-pruned",
        help="Path to save the pruned model",
    )

    return parser.parse_args()


def get_dataloader(args, tokenizer):
    """Create calibration dataloader for pruning.

    Args:
        args: Parsed command-line arguments
        tokenizer: Tokenizer for the model

    Returns:
        DataLoader: Calibration data loader
    """
    return get_calib_dataloder(
        dataset=args.dataset_name_or_path,
        tokenizer=tokenizer,
        max_block_size=2048,
        n_blocks_for_stat=args.max_steps,
        batch_size=1,
        num_workers=4,
    )


def evaluation(args, model, tokenizer):
    """Evaluate pruned model on perplexity and zero-shot tasks.

    Evaluates the model on:
    - Perplexity: wikitext2, c4
    - Zero-shot tasks: arc_challenge, arc_easy, boolq, piqa, winogrande,
      hellaswag, mmlu(hendrycksTest), openbookqa

    Results are saved to a file and printed to console.

    Args:
        args: Parsed command-line arguments
        model: Pruned model to evaluate
        tokenizer: Tokenizer for the model
    """
    model.eval()
    model.seqlen = 2048

    # Evaluation datasets
    ppl_datasets = ["wikitext2","c4"]
    zeroshot_datasets = [
        ["arc_challenge"],
        ["arc_easy"],
        ["boolq"],
        ["piqa"],
        ["winogrande"],
        ["hellaswag"],
        ["hendrycksTest*"],
        ["openbookqa"],
    ]

    # Create results filename
    results_file = f"{args.expert_prune_metric}_{args.preserve_n_experts}_{args.tau}.txt"

    with open(results_file, "a+", encoding="utf-8") as f:
        # Evaluate perplexity
        for dataset in ppl_datasets:
            ppl = eval_ppl(
                dataset, model, tokenizer, device=torch.device("cuda:0")
            )
            result = f"\nPPL on {dataset}: {ppl:.4f}\n"
            f.write(result)
            print(result)

        # Evaluate zero-shot tasks
        for dataset in zeroshot_datasets:
            results = eval_zero_shot(
                args.model_name_or_path, model, tokenizer, task_list=dataset
            )
            acc = results["results"][dataset[0]]["acc"]
            f.write(f"{acc:.4f}\n")
            print(f"{dataset[0]} accuracy: {acc:.4f}")


def main():
    """Main entry point for MoE expert pruning."""
    # Parse command-line arguments
    args = parse_args()

    # Load model and tokenizer
    print(f"Loading model from {args.model_name_or_path}...")
    model, tokenizer = prepare_model_and_tokenizer(
        args.model_name_or_path, mode="prune"
    )
    model.eval()

    # Create calibration dataloader
    print("Loading calibration dataset...")
    train_dataloader = get_dataloader(args, tokenizer)

    # Perform expert pruning
    if args.expert_prune:
        print(f"Pruning experts using {args.expert_prune_metric} metric...")
        expert_prune(args, model, train_dataloader, tokenizer)
        print("Pruning completed.")

    # Evaluate the pruned model
    # print("Evaluating pruned model...")
    # evaluation(args, model, tokenizer)
    # print("Evaluation completed.")


if __name__ == "__main__":
    main()
