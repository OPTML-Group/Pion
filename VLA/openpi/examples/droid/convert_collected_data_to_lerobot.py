"""Convert self-collected DROID-format data (data/collected_data/<task>/<episode>/...)
into LeRobot datasets, one dataset per task.

Each <episode> folder is expected to follow the DROID raw layout:
    <episode>/
      ├── trajectory.h5            (with action_info/joint_velocity, gripper_position, robot_state)
      ├── metadata_*.json
      └── recordings/MP4/<cam_id>.mp4   (3 cameras: 1 wrist + 2 exterior)

The wrist / exterior cameras are identified using observation/camera_type in the HDF5 file
(0 = hand_camera, non-zero = varied_camera).

Usage:
    uv run examples/droid/convert_collected_data_to_lerobot.py \
        --data_dir <DATA_ROOT>/collected_data \
        --task_name cubic_to_bowl \
        --repo_name your-hf-username/cubic_to_bowl

The resulting dataset is saved under $LEROBOT_HOME / <repo_name>.

Action space: 7D joint velocity + 1D gripper position = 8D, identical to pi05-droid pretraining.
"""

from __future__ import annotations

import copy
import dataclasses
import glob
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from PIL import Image
from tqdm import tqdm


TASK_PROMPTS: dict[str, str] = {
    "cubic_to_bowl": "put the rubik's cube into the bowl",
    "cubic_to_plate": "put the rubik's cube on the plate",
    "cucumber_to_plate": "put the cucumber on the plate",
}


def resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(image)
    return np.array(image.resize(size, resample=Image.BICUBIC))


# ──────────────────────────────────────────────────────────────────────────────
# Lightweight DROID raw-data reader (subset of the upstream r2d2 reader, just
# what we need to extract per-step images + robot state + actions).
# ──────────────────────────────────────────────────────────────────────────────


CAMERA_TYPE_HAND = 0  # observation/camera_type/<cam_id> == 0  → wrist


class MP4Reader:
    """Reads frames from an MP4 in order; rewinds when needed."""

    def __init__(self, filepath: str):
        self._reader = cv2.VideoCapture(filepath)
        if not self._reader.isOpened():
            raise RuntimeError(f"Cannot open MP4 file: {filepath}")
        self._index = 0

    def read_frame(self, target_index: int) -> np.ndarray | None:
        if target_index < self._index:
            self._reader.set(cv2.CAP_PROP_POS_FRAMES, target_index)
            self._index = target_index
        while self._index < target_index:
            ok, _ = self._reader.read()
            if not ok:
                return None
            self._index += 1
        ok, frame = self._reader.read()
        if not ok:
            return None
        self._index += 1
        # OpenCV returns BGR; convert to RGB
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self) -> None:
        self._reader.release()


def load_episode(episode_dir: Path):
    """Yield per-step dicts {wrist, ext1, ext2, joint_position, gripper_position, action}."""
    h5_path = episode_dir / "trajectory.h5"
    mp4_dir = episode_dir / "recordings" / "MP4"
    if not h5_path.exists() or not mp4_dir.exists():
        return None

    with h5py.File(h5_path, "r") as f:
        joint_positions = f["observation/robot_state/joint_positions"][:]      # (T, 7)
        gripper_positions = f["observation/robot_state/gripper_position"][:]   # (T,)
        joint_velocity = f["action_info/joint_velocity"][:]                    # (T, 7)
        action_gripper = f["action_info/gripper_position"][:]                  # (T,)
        camera_types = {
            cam_id: f[f"observation/camera_type/{cam_id}"][:]                  # (T,)
            for cam_id in f["observation/camera_type"].keys()
        }

    T = joint_positions.shape[0]

    # Identify wrist / exterior cams from camera_type[0]
    wrist_ids = [c for c, ct in camera_types.items() if int(ct[0]) == CAMERA_TYPE_HAND]
    exterior_ids = [c for c, ct in camera_types.items() if int(ct[0]) != CAMERA_TYPE_HAND]

    if len(wrist_ids) < 1 or len(exterior_ids) < 2:
        print(f"  ! Skipping {episode_dir.name}: need 1 wrist + 2 exterior cams, "
              f"found wrist={wrist_ids}, exterior={exterior_ids}")
        return None

    wrist_id = wrist_ids[0]
    ext1_id, ext2_id = exterior_ids[0], exterior_ids[1]

    wrist_mp4 = mp4_dir / f"{wrist_id}.mp4"
    ext1_mp4 = mp4_dir / f"{ext1_id}.mp4"
    ext2_mp4 = mp4_dir / f"{ext2_id}.mp4"
    for p in (wrist_mp4, ext1_mp4, ext2_mp4):
        if not p.exists():
            print(f"  ! Skipping {episode_dir.name}: missing {p.name}")
            return None

    wrist_reader = MP4Reader(str(wrist_mp4))
    ext1_reader = MP4Reader(str(ext1_mp4))
    ext2_reader = MP4Reader(str(ext2_mp4))

    try:
        for t in range(T):
            wrist_frame = wrist_reader.read_frame(t)
            ext1_frame = ext1_reader.read_frame(t)
            ext2_frame = ext2_reader.read_frame(t)
            if wrist_frame is None or ext1_frame is None or ext2_frame is None:
                break
            yield {
                "wrist": wrist_frame,
                "ext1": ext1_frame,
                "ext2": ext2_frame,
                "joint_position": np.asarray(joint_positions[t], dtype=np.float32),
                "gripper_position": np.asarray([gripper_positions[t]], dtype=np.float32),
                "action": np.concatenate(
                    [joint_velocity[t], np.asarray([action_gripper[t]])],
                    dtype=np.float32,
                ),
            }
    finally:
        wrist_reader.release()
        ext1_reader.release()
        ext2_reader.release()


# ──────────────────────────────────────────────────────────────────────────────
# Main conversion
# ──────────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Args:
    data_dir: str
    task_name: str
    repo_name: str | None = None  # default: "collected_data/<task_name>"
    push_to_hub: bool = False
    fps: int = 15
    image_size: tuple[int, int] = (320, 180)  # (W, H), matches DROID RLDS


def main(args: Args) -> None:
    data_root = Path(args.data_dir).resolve()
    task_dir = data_root / args.task_name
    if not task_dir.exists():
        raise FileNotFoundError(f"Task folder not found: {task_dir}")
    if args.task_name not in TASK_PROMPTS:
        raise ValueError(
            f"Unknown task '{args.task_name}'. Add it to TASK_PROMPTS or pass a known name. "
            f"Known: {list(TASK_PROMPTS.keys())}"
        )
    prompt = TASK_PROMPTS[args.task_name]

    repo_name = args.repo_name or f"collected_data/{args.task_name}"
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    W, H = args.image_size
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=args.fps,
        features={
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (H, W, 3),
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": {
                "dtype": "image",
                "shape": (H, W, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (H, W, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {"dtype": "float32", "shape": (7,), "names": ["joint_position"]},
            "gripper_position": {"dtype": "float32", "shape": (1,), "names": ["gripper_position"]},
            "actions": {"dtype": "float32", "shape": (8,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    episode_dirs = sorted(p for p in task_dir.iterdir() if p.is_dir())
    print(f"Task '{args.task_name}': found {len(episode_dirs)} episodes")
    print(f"  Language instruction: \"{prompt}\"")

    n_ok, n_skip = 0, 0
    for episode_dir in tqdm(episode_dirs, desc=f"Converting {args.task_name}"):
        steps = load_episode(episode_dir)
        if steps is None:
            n_skip += 1
            continue

        wrote_any = False
        for step in steps:
            dataset.add_frame(
                {
                    "exterior_image_1_left": resize_image(step["ext1"], (W, H)),
                    "exterior_image_2_left": resize_image(step["ext2"], (W, H)),
                    "wrist_image_left": resize_image(step["wrist"], (W, H)),
                    "joint_position": step["joint_position"],
                    "gripper_position": step["gripper_position"],
                    "actions": step["action"],
                    "task": prompt,
                }
            )
            wrote_any = True

        if wrote_any:
            dataset.save_episode()
            n_ok += 1
        else:
            n_skip += 1

    print(f"Done: wrote {n_ok} episodes, skipped {n_skip}")
    print(f"Dataset saved to: {output_path}")

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["droid", "panda", "collected"], private=False, push_videos=True, license="apache-2.0"
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
