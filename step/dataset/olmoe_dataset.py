import os
import io
import json
import glob
from pathlib import Path
import datasets
from datasets import load_dataset, Dataset, IterableDatasetDict, IterableDataset, Features, Value
import zstandard as zstd
import gzip
from typing import Dict, List, Optional, Tuple, Union, Iterator, Any
import logging
import random
from collections import defaultdict
import numpy as np
from itertools import cycle, islice
import torch
from torch.utils.data import get_worker_info
from datasets.utils.py_utils import size_str
from datasets.iterable_dataset import ExamplesIterable

logger = logging.getLogger(__name__)

_CITATION = """
@misc{muennighoff2024olmoeopenmixtureofexpertslanguage,
      title={OLMoE: Open Mixture-of-Experts Language Models}, 
      author={Niklas Muennighoff and Luca Soldaini and Dirk Groeneveld and Kyle Lo and Jacob Morrison and Sewon Min and Weijia Shi and Pete Walsh and Oyvind Tafjord and Nathan Lambert and Yuling Gu and Shane Arora and Akshita Bhagia and Dustin Schwenk and David Wadden and Alexander Wettig and Binyuan Hui and Tim Dettmers and Douwe Kiela and Ali Farhadi and Noah A. Smith and Pang Wei Koh and Amanpreet Singh and Hannaneh Hajishirzi},
      year={2024},
      eprint={2409.02060},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.02060}, 
}
"""

_DESCRIPTION = """
OLMoE Mix dataset is a large-scale dataset used to train OLMoE-1B-7B, a Mixture-of-Experts LLM 
with 1B active and 7B total parameters. The dataset contains text from various sources including
DCLM Baseline, Starcoder, peS2o, Arxiv, OpenWebMath, Algebraic Stack, and Wikipedia.
"""

# Define file patterns for different parts of the dataset
DATASET_PATTERNS = {
    "train": "**/*.json.*",  # Both json.zst and json.gz files
}

class OLMoEMixDatasetBuilder(datasets.GeneratorBasedBuilder):
    """OLMoE Mix dataset for language model pretraining."""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="default",
            version=VERSION,
            description="OLMoE Mix dataset for language model pretraining",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "id": datasets.Value("string"),
                    "added": datasets.Value("string"),
                    "created": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://huggingface.co/datasets/allenai/OLMoE-mix-0924",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = os.environ.get("OLMOE_DATA_DIR", None)
        if data_dir is None:
            raise ValueError(
                "Environment variable OLMOE_DATA_DIR must be set to the directory containing the OLMoE mix data files."
            )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_dir": data_dir, "split": "train"},
            ),
        ]

    def _read_zst_file(self, filepath):
        """Read a zstandard compressed JSON file."""
        with open(filepath, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                text = reader.read().decode("utf-8")
                for line in text.splitlines():
                    if line.strip():
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            logger.warning(f"Could not decode JSON line in {filepath}")

    def _read_gz_file(self, filepath):
        """Read a gzip compressed JSON file."""
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(f"Could not decode JSON line in {filepath}")

    def _generate_examples(self, data_dir, split):
        """Yields examples."""
        pattern = DATASET_PATTERNS[split]
        files = glob.glob(os.path.join(data_dir, pattern), recursive=True)
        
        if not files:
            logger.warning(f"No matching files found in {data_dir} with pattern {pattern}")
            return

        logger.info(f"Found {len(files)} files for {split} split")
        
        for file_id, filepath in enumerate(files):
            logger.info(f"Processing file {filepath}")
            
            if filepath.endswith(".json.zst"):
                examples = self._read_zst_file(filepath)
            elif filepath.endswith(".json.gz"):
                examples = self._read_gz_file(filepath)
            else:
                logger.warning(f"Unsupported file format: {filepath}")
                continue
            
            for i, example in enumerate(examples):
                # Ensure the example has a text field 
                if "text" not in example:
                    # Handle different possible JSON structures
                    if "content" in example:
                        example["text"] = example["content"]
                    else:
                        # Skip examples without text content
                        continue
                
                # Ensure we have an ID field
                if "id" not in example:
                    example["id"] = f"{file_id}-{i}"
                
                # Add other required fields if missing
                if "added" not in example:
                    example["added"] = ""
                if "created" not in example:
                    example["created"] = ""
                
                yield f"{file_id}-{i}", {
                    "text": example["text"],
                    "id": example["id"],
                    "added": example["added"],
                    "created": example["created"]
                }

class OLMoEExamplesIterable(ExamplesIterable):
    """Custom ExamplesIterable for OLMoE dataset"""
    
    def __init__(
        self,
        base_dir: str,
        buffer_size: int,
        features: Features,
        source_props: Dict[str, float],
        seed: Optional[int] = None
    ):
        # Initialize ExamplesIterable with an empty generator and features
        super().__init__(iter([]), features)
        self.base_dir = base_dir
        self.buffer_size = buffer_size
        self.source_props = source_props
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self._example_idx = 0  # Counter for generating unique keys
    
    def _get_iterator_for_worker(
        self,
        worker_id: Optional[int],
        num_workers: Optional[int]
    ) -> Iterator[Tuple[str, Dict[str, Any]]]:  # Note the return type change
        """Get iterator for specific worker in distributed setting"""
        # Set worker-specific seed
        if worker_id is not None:
            worker_seed = self.seed + worker_id
            random.seed(worker_seed)
            self._example_idx = worker_id * 1000000  # Offset for each worker
        
        # Initialize buffers for each source
        buffers = {source: [] for source in self.source_props.keys()}
        
        # Create file iterators for each source
        iterators = {}
        for source in self.source_props.keys():
            source_dir = os.path.join(self.base_dir, source)
            files = glob.glob(os.path.join(source_dir, "**/*.json.*"), recursive=True)
            
            if num_workers and worker_id is not None:
                # Distribute files among workers
                files = files[worker_id::num_workers]
            
            if not files:
                print(f"Warning: No files found in {source_dir}")
                continue
                
            random.shuffle(files)
            iterators[source] = self._create_file_iterator(files)
        
        # Create infinite iterator that yields mixed examples
        while True:
            # Sample source according to proportions
            available_sources = [s for s in self.source_props.keys() if s in iterators]
            if not available_sources:
                raise ValueError("No valid source directories found with data files")
                
            source_weights = [self.source_props[s] for s in available_sources]
            source = random.choices(
                available_sources,
                weights=source_weights,
                k=1
            )[0]
            
            # Fill buffer if needed
            if len(buffers[source]) < 1:
                try:
                    while len(buffers[source]) < self.buffer_size:
                        item = next(iterators[source])
                        item["source"] = source  # Add source information
                        buffers[source].append(item)
                    random.shuffle(buffers[source])
                except StopIteration:
                    # Reinitialize iterator if exhausted
                    files = glob.glob(os.path.join(self.base_dir, source, "**/*.json.*"), recursive=True)
                    if num_workers and worker_id is not None:
                        files = files[worker_id::num_workers]
                    random.shuffle(files)
                    iterators[source] = self._create_file_iterator(files)
                    continue
            
            # Yield next item from buffer with a unique key
            example = buffers[source].pop()
            key = f"{source}-{self._example_idx}"
            self._example_idx += 1
            yield key, example
    
    def _create_file_iterator(self, files: List[str]) -> Iterator[Dict[str, Any]]:
        """Create an iterator over files"""
        for file in cycle(files):  # Use cycle for infinite iteration
            if file.endswith('.json.zst'):
                yield from self._read_zst_file(file)
            elif file.endswith('.json.gz'):
                yield from self._read_gz_file(file)
    
    def _read_zst_file(self, filepath: str) -> Iterator[Dict[str, Any]]:
        """Read a zstandard compressed JSON file"""
        try:
            with open(filepath, "rb") as f:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f) as reader:
                    for line in io.TextIOWrapper(reader, encoding='utf-8'):
                        if line.strip():
                            try:
                                example = json.loads(line)
                                # Ensure all required fields are present
                                if "text" not in example and "content" in example:
                                    example["text"] = example["content"]
                                if "id" not in example:
                                    example["id"] = ""
                                if "added" not in example:
                                    example["added"] = ""
                                if "created" not in example:
                                    example["created"] = ""
                                yield example
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            print(f"Error reading file {filepath}: {str(e)}")
            return  # Skip this file on error
    
    def _read_gz_file(self, filepath: str) -> Iterator[Dict[str, Any]]:
        """Read a gzip compressed JSON file"""
        try:
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            example = json.loads(line)
                            # Ensure all required fields are present
                            if "text" not in example and "content" in example:
                                example["text"] = example["content"]
                            if "id" not in example:
                                example["id"] = ""
                            if "added" not in example:
                                example["added"] = ""
                            if "created" not in example:
                                example["created"] = ""
                            yield example
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"Error reading file {filepath}: {str(e)}")
            return  # Skip this file on error

    def __iter__(self):
        """Return iterator over the dataset"""
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else None
        num_workers = worker_info.num_workers if worker_info else None
        
        return self._get_iterator_for_worker(worker_id, num_workers)

class ProportionalMixIterableDataset(IterableDataset):
    """
    A HuggingFace IterableDataset that ensures proportional sampling across different data sources
    while maintaining good shuffling properties. Compatible with streaming and accelerator.
    """
    def __init__(
        self, 
        base_dir: str, 
        buffer_size: int = 100_000,
        streaming: bool = True,
        seed: Optional[int] = None
    ):
        """
        Args:
            base_dir: Directory containing the dataset files
            buffer_size: Size of the buffer for each category for shuffling
            streaming: Whether to use streaming mode
            seed: Random seed for reproducibility
        """
        # Define source directories and their target proportions based on token counts
        source_props = {
            "dclm": 0.9537481561280791,    # DCLM Baseline
            "starcoder": 0.024955586468636267,  # 0.15, Starcoder 
            "pes2o": 0.01413326283174252,    # peS2o
            "algebraic-stack": 0.0031132711832160096,    # 0.10, Algebraic Stack
            "open-web-math": 0.0031379796846701044,     # 0.10, OpenWebMath
            "wiki": 0.000911743703656117      # Wikipedia + Wikibooks
        }
        
        features = Features({
            "text": Value("string"),
            "id": Value("string"),
            "added": Value("string"),
            "created": Value("string"),
            "source": Value("string")  # Added source field
        })

        info = datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "id": datasets.Value("string"),
                    "added": datasets.Value("string"),
                    "created": datasets.Value("string"),
                    "source": Value("string")  # Added source field
                }
            ),
            supervised_keys=None,
            homepage="https://huggingface.co/datasets/allenai/OLMoE-mix-0924",
            citation=_CITATION,
        )
        
        # Initialize the parent class with our custom iterable
        super().__init__(
            OLMoEExamplesIterable(
                base_dir=base_dir,
                buffer_size=buffer_size,
                features=features,
                source_props=source_props,
                seed=seed,
            ),
            info=info,
        )
        
        self.base_dir = base_dir
        self.buffer_size = buffer_size
        self.streaming = streaming
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self.source_props = source_props
        self._features = features

    @property
    def features(self):
        """Return the features of the dataset"""
        return self._features

def load_olmoe_mix_dataset(
    data_dir: str,
    buffer_size: int = 100_000,
    streaming: bool = True,
    seed: Optional[int] = None
) -> IterableDataset:
    """
    Load the OLMoE mix dataset with improved mixing strategy.
    
    Args:
        data_dir: Directory containing the OLMoE mix data files
        buffer_size: Size of the buffer for each category for shuffling
        streaming: Whether to use streaming mode
        seed: Random seed for reproducibility
    
    Returns:
        A HuggingFace IterableDataset that provides well-mixed examples
    """
    train_dataset = ProportionalMixIterableDataset(
        base_dir=data_dir,
        buffer_size=buffer_size,
        streaming=streaming,
        seed=seed
    ) 
    return datasets.IterableDatasetDict({
        "train": train_dataset
    })

def test_dataset_loading(data_dir: str):
    """
    Test function to verify if the dataset loads successfully.
    
    Args:
        data_dir: Directory containing the OLMoE mix data files
    """
    try:
        print(f"Attempting to load dataset from {data_dir}")
        
        # Check if directory exists
        if not os.path.exists(data_dir):
            raise ValueError(f"Directory does not exist: {data_dir}")
            
        # Check for required subdirectories and files
        #required_dirs = ["dclm", "starcoder", "pes2o", "algebraic-stack", "open-web-math", "wiki"]
        required_dirs = ["dclm", "starcoder", "pes2o", "algebraic-stack", "open-web-math", "wiki"]
        dir_stats = {}
        
        print("\nChecking directories and files:")
        for d in required_dirs:
            dir_path = os.path.join(data_dir, d)
            if not os.path.exists(dir_path):
                print(f"- {d}: Directory missing")
                continue
                
            files = glob.glob(os.path.join(dir_path, "**/*.json.*"), recursive=True)
            dir_stats[d] = len(files)
            print(f"- {d}: Found {len(files)} files")
        
        if not any(dir_stats.values()):
            raise ValueError("No data files found in any directory")
        
        print("\nAvailable directories:", [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        
        # Try loading the dataset
        print("\nInitializing dataset...")
        dataset = load_olmoe_mix_dataset(data_dir)
        
        # Try to get a few examples
        print("\nTrying to get first few examples...")
        source_counts = defaultdict(int)
        
        for i, example in enumerate(dataset["train"]):
            print(f"\nExample {i + 1}:")
            print(f"Key: {example['id']}")  # Print the key
            print(f"Source: {example['source']}")
            print(f"Text preview: {example['text'][:100]}...")
            source_counts[example['source']] += 1
            
            if i >= 9:  # Show first 10 examples
                break
        
        print("\nSource distribution in sample:")
        for source, count in source_counts.items():
            print(f"{source}: {count} examples")
            
        print("\nDataset loaded and iterated successfully!")
        return True
        
    except Exception as e:
        print(f"\nError loading dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python olmoe_dataset.py <data_directory>")
        sys.exit(1)
        
    data_dir = sys.argv[1]
    success = test_dataset_loading(data_dir)
    sys.exit(0 if success else 1) 
