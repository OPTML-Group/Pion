"""
Preprocess the BytedTsinghua-SIA/DAPO-Math-17k dataset to parquet format.
DAPO-Math-17k is a 17k-problem training set from the DAPO paper (arxiv:2503.14476).

The HuggingFace dataset already includes verl-compatible fields:
  data_source, prompt, reward_model, extra_info, ability

Usage:
    python examples/data_preprocess/dapo_math_17k.py --local_save_dir ~/data/dapo_math_17k
"""

import argparse
import json
import os

import datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_save_dir",
        default="~/data/dapo_math_17k",
        help="The save directory for the preprocessed dataset.",
    )
    args = parser.parse_args()

    data_source = "BytedTsinghua-SIA/DAPO-Math-17k"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, "default")

    train_dataset = dataset["train"]
    print(f"DAPO-Math-17k train set: {len(train_dataset)} examples")

    local_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))

    example = train_dataset[0]
    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(example, f, indent=2)

    print(f"Saved {len(train_dataset)} examples to {local_dir}/train.parquet")
