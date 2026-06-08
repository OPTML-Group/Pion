"""
Preprocess MATH dataset to multi-turn SFT parquet format with `messages` field.

Usage:
    MATH_MIN_LEVEL=3 MATH_MAX_LEVEL=5 \
    python examples/data_preprocess/math_multiturn_sft.py --local_save_dir ~/data/math_level3-5_sft
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/math_level3-5_sft", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = "DigitalLearningGmbH/MATH-lighteval"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Filter by difficulty level (e.g., Level 3-5 as in Dr. GRPO paper)
    min_level = int(os.environ.get("MATH_MIN_LEVEL", 3))
    max_level = int(os.environ.get("MATH_MAX_LEVEL", 5))
    if min_level > 1 or max_level < 5:
        level_filter = {f"Level {i}" for i in range(min_level, max_level + 1)}
        train_dataset = train_dataset.filter(lambda x: x["level"] in level_filter)
        test_dataset = test_dataset.filter(lambda x: x["level"] in level_filter)
        print(f"Filtered to Level {min_level}-{max_level}: {len(train_dataset)} train, {len(test_dataset)} test")

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    def make_map_fn():
        def process_fn(example, idx):
            question = example["problem"] + " " + instruction_following
            answer = example["solution"]
            return {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                "extra_info": {
                    "index": idx,
                    "subject": example.get("subject", ""),
                    "level": example.get("level", ""),
                },
            }

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn(), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn(), with_indices=True)

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
    print(f"Saved SFT dataset to: {local_save_dir}")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
