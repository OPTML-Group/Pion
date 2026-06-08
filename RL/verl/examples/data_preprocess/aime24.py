"""
Preprocess the BytedTsinghua-SIA/AIME-2024 dataset to parquet format.

The HuggingFace dataset already has verl-compatible fields (data_source, prompt,
ability, reward_model, extra_info), so we just download and save as parquet.

Usage:
    python examples/data_preprocess/aime24.py --local_save_dir ~/data/aime24
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/aime24", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()

    data_source = "BytedTsinghua-SIA/AIME-2024"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path, "default")
    else:
        dataset = datasets.load_dataset(data_source, "default")

    train_dataset = dataset["train"]
    print(f"AIME-2024: {len(train_dataset)} problems")

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_dir, exist_ok=True)
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    example = train_dataset[0]
    with open(os.path.join(local_dir, "test_example.json"), "w") as f:
        json.dump(example, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(train_dataset)} problems to {local_dir}/test.parquet")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
