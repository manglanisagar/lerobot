import os
import pandas as pd
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import json

INPUT_DIR = "dataset"
OUTPUT_DIR = "processed_dataset"
PROMPTS_FILE = os.path.join(INPUT_DIR, "prompts.csv")
IMAGE_SIZE = (640, 360)  # width, height
KEEP_RANGE = range(50, 230)  # 50 to 229 inclusive
FPS = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load prompts
prompt_df = pd.read_csv(PROMPTS_FILE).set_index("scene_index")

metadata = {
    "codebase_version": "v2.1",
    "robot_type": "custom_quad",
    "total_episodes": 0,
    "total_frames": 0,
    "fps": FPS,
    "features": {
        "observation.images.perspective": {
            "dtype": "video",
            "shape": [360, 640, 3],
            "names": ["height", "width", "channels"]
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [3],
            "names": ["vx", "vy", "yaw"]
        },
        "action": {
            "dtype": "float32",
            "shape": [3],
            "names": ["vx", "vy", "omega"]
        },
        "language_instruction": {
            "dtype": "string",
            "shape": [1],
            "names": None
        },
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]}
    }
}

episode_count = 0
frame_count = 0

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

    episode_data = {
        "observation.images.perspective": [],
        "observation.state": [],
        "action": [],
        "language_instruction": [],
        "timestamp": [],
        "frame_index": [],
        "episode_index": []
    }

    for i in KEEP_RANGE:
        img_path = os.path.join(scene_path, f"rgb_{i:04d}.png")
        img = cv2.imread(img_path)
        img = cv2.resize(img, IMAGE_SIZE)
        img = img[..., ::-1]  # Convert BGR to RGB
        episode_data["observation.images.perspective"].append(img)

        obs_row = vel_df.iloc[i]
        cmd_row = cmd_df.iloc[i]

        state = [obs_row["vx"], obs_row["vy"], obs_row["wz"]]
        action = [cmd_row["vx"], cmd_row["vy"], cmd_row["omega"]]

        episode_data["observation.state"].append(state)
        episode_data["action"].append(action)
        episode_data["language_instruction"].append([prompt])
        episode_data["timestamp"].append([obs_row["time"]])
        episode_data["frame_index"].append([i])
        episode_data["episode_index"].append([episode_count])

    episode_data["observation.images.perspective"] = [
        cv2.imencode('.jpg', img)[1].tobytes()
        for img in episode_data["observation.images.perspective"]
    ]

    table = pa.Table.from_pydict(episode_data)
    pq.write_table(
        table,
        os.path.join(OUTPUT_DIR, f"episode_{episode_count:06d}.parquet")
    )

    frame_count += len(episode_data["frame_index"])
    episode_count += 1

metadata["total_episodes"] = episode_count
metadata["total_frames"] = frame_count

with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

