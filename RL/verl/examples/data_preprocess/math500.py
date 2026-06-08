"""
Preprocess the HuggingFaceH4/MATH-500 dataset to parquet format.
MATH-500 is a 500-problem test subset commonly used for evaluation.

Usage:
    python examples/data_preprocess/math500.py --local_save_dir ~/data/math500
"""

import argparse
import json
import os

import datasets

from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_save_dir", default="~/data/math500", help="The save directory for the preprocessed dataset."
    )
    args = parser.parse_args()

    data_source = "HuggingFaceH4/MATH-500"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source)

    test_dataset = dataset["test"]
    print(f"MATH-500 test set: {len(test_dataset)} examples")

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    def process_fn(example, idx):
        question = example["problem"] + " " + instruction_following
        solution = extract_solution(example["solution"])
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {"split": "test", "index": idx, "subject": example.get("subject", ""), "level": example.get("level", "")},
        }

    test_dataset = test_dataset.map(function=process_fn, with_indices=True)

    local_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_dir, exist_ok=True)

    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    example = test_dataset[0]
    with open(os.path.join(local_dir, "test_example.json"), "w") as f:
        json.dump(example, f, indent=2)

    print(f"Saved to {local_dir}/test.parquet")
