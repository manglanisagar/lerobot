import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from lerobot.common.datasets.compute_stats import get_feature_stats
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_FEATURES,
    DEFAULT_PARQUET_PATH,
    create_empty_dataset_info,
    write_episode,
    write_episode_stats,
    write_info,
    write_task,
)

INPUT_DIR = "dataset"
OUTPUT_DIR = "processed_dataset"
PROMPTS_FILE = os.path.join(INPUT_DIR, "prompts.csv")
IMAGE_SIZE = (640, 360)  # width, height
IGNORE_FRAMES = 50
MIN_STOP_INDEX = 150
STOP_VEL_THRESHOLD = 0.01
FPS = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load prompts
prompt_df = pd.read_csv(PROMPTS_FILE).set_index("scene_index")

features = {
    "observation.images.perspective": {
        "dtype": "image",
        "shape": [360, 640, 3],
        "names": ["height", "width", "channels"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": [3],
        "names": ["vx", "vy", "yaw"],
    },
    "action": {
        "dtype": "float32",
        "shape": [3],
        "names": ["vx", "vy", "omega"],
    },
    "next.reward": {"dtype": "float32", "shape": [1], "names": None},
    "next.done": {"dtype": "bool", "shape": [1], "names": None},
}
features = {**features, **DEFAULT_FEATURES}

metadata = create_empty_dataset_info(
    CODEBASE_VERSION,
    FPS,
    features,
    use_videos=False,
    robot_type="custom_rl_robot",
)

output_root = Path(OUTPUT_DIR)
episode_count = 0
frame_count = 0
global_index = 0
tasks_to_index: dict[str, int] = {}

for scene in tqdm(sorted(os.listdir(INPUT_DIR))):
    scene_path = os.path.join(INPUT_DIR, scene)
    if not os.path.isdir(scene_path):
        continue

    # Parse scene index from name like scene_0000
    try:
        scene_index = int(scene.split("_")[1])
        prompt = prompt_df.loc[scene_index]["prompt"]
    except Exception as e:
        print(f"Skipping scene {scene} due to missing or invalid prompt: {e}")
        continue

    vel_df = pd.read_csv(os.path.join(scene_path, "velocity.csv"))
    cmd_df = pd.read_csv(os.path.join(scene_path, "commands.csv"))

    # Determine success frame index
    success_idx = None
    for i in range(MIN_STOP_INDEX, len(vel_df)):
        speed = (vel_df.loc[i, "vx"] ** 2 + vel_df.loc[i, "vy"] ** 2) ** 0.5
        if speed < STOP_VEL_THRESHOLD:
            success_idx = i
            break
    if success_idx is None:
        success_idx = len(vel_df) - 1

    if prompt not in tasks_to_index:
        task_idx = len(tasks_to_index)
        tasks_to_index[prompt] = task_idx
        write_task(task_idx, prompt, output_root)
    else:
        task_idx = tasks_to_index[prompt]

    episode_data = {
        "observation.images.perspective": [],
        "observation.state": [],
        "action": [],
        "next.reward": [],
        "next.done": [],
        "timestamp": [],
        "frame_index": [],
        "episode_index": [],
        "task_index": [],
        "index": [],
    }

    images_for_stats = []
    states_for_stats = []
    actions_for_stats = []

    for i in range(IGNORE_FRAMES, success_idx + 1):
        img_path = os.path.join(scene_path, f"rgb_{i:04d}.png")
        img = cv2.imread(img_path)
        img = cv2.resize(img, IMAGE_SIZE)
        img = img[..., ::-1]  # Convert BGR to RGB
        episode_data["observation.images.perspective"].append(img)
        images_for_stats.append(img.transpose(2, 0, 1))

        obs_row = vel_df.iloc[i]
        cmd_row = cmd_df.iloc[i]

        state = [obs_row["vx"], obs_row["vy"], obs_row["yaw"]]
        action = [cmd_row["vx"], cmd_row["vy"], cmd_row["omega"]]

        done = i == success_idx
        reward = 1.0 if done else -0.001 - 0.001 * (cmd_row["vx"] + cmd_row["vy"] + obs_row["wz"])

        episode_data["observation.state"].append(state)
        episode_data["action"].append(action)
        episode_data["next.reward"].append([reward])
        episode_data["next.done"].append([done])
        states_for_stats.append(state)
        actions_for_stats.append(action)
        episode_data["timestamp"].append([obs_row["time"]])
        episode_data["frame_index"].append([i])
        episode_data["episode_index"].append([episode_count])
        episode_data["task_index"].append([task_idx])
        episode_data["index"].append([global_index])
        global_index += 1

    episode_data["observation.images.perspective"] = [
        cv2.imencode(".jpg", img)[1].tobytes() for img in episode_data["observation.images.perspective"]
    ]

    ep_stats_buffer = {
        "observation.images.perspective": np.stack(images_for_stats).astype(np.float32) / 255.0,
        "observation.state": np.stack(states_for_stats).astype(np.float32),
        "action": np.stack(actions_for_stats).astype(np.float32),
        "timestamp": np.array(episode_data["timestamp"], dtype=np.float32).squeeze(),
        "frame_index": np.array(episode_data["frame_index"], dtype=np.int64).squeeze(),
        "episode_index": np.array(episode_data["episode_index"], dtype=np.int64).squeeze(),
        "task_index": np.array(episode_data["task_index"], dtype=np.int64).squeeze(),
        "index": np.array(episode_data["index"], dtype=np.int64).squeeze(),
    }

    ep_stats = {
        key: get_feature_stats(
            value,
            axis=(0, 2, 3) if key == "observation.images.perspective" else 0,
            keepdims=key == "observation.images.perspective",
        )
        for key, value in ep_stats_buffer.items()
    }
    ep_stats["observation.images.perspective"] = {
        k: v if k == "count" else np.squeeze(v, axis=0)
        for k, v in ep_stats["observation.images.perspective"].items()
    }

    table = pa.Table.from_pydict(episode_data)
    ep_path = output_root / DEFAULT_PARQUET_PATH.format(
        episode_chunk=episode_count // DEFAULT_CHUNK_SIZE,
        episode_index=episode_count,
    )
    ep_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, ep_path)

    write_episode(
        {
            "episode_index": episode_count,
            "tasks": [prompt],
            "length": len(range(IGNORE_FRAMES, success_idx + 1)),
        },
        output_root,
    )
    write_episode_stats(episode_count, ep_stats, output_root)

    frame_count += len(episode_data["frame_index"])
    episode_count += 1

metadata["total_episodes"] = episode_count
metadata["total_frames"] = frame_count
metadata["total_tasks"] = len(tasks_to_index)
metadata["total_chunks"] = (episode_count - 1) // DEFAULT_CHUNK_SIZE + 1

write_info(metadata, output_root)
