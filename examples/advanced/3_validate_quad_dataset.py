import math
from pathlib import Path

import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.policies.factory import get_policy_class
from lerobot.configs.policies import PreTrainedConfig


def main():
    device = torch.device("cuda")

    # Path to the trained policy
    pretrained_policy_path = Path("outputs/train/2025-06-09/00-18-15_smolvla/checkpoints/005000/pretrained_model")

    cfg = PreTrainedConfig.from_pretrained(pretrained_policy_path, local_files_only=True)
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(pretrained_policy_path, local_files_only=True)
    policy.eval()
    policy.to(device)

    # Setup dataset and split episodes
    dataset_root = Path("quad_data/processed_dataset")
    repo_id = "quad_data/processed_dataset"

    dataset_metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    fps = dataset_metadata.fps

    # Compute the delta timestamps expected by the policy. This ensures that the
    # sequence length of the data matches what the model was trained with and
    # avoids mismatches when constructing the attention masks.
    delta_timestamps = {
        "observation.images.perspective": [i / fps for i in cfg.observation_delta_indices],
        "observation.state": [i / fps for i in cfg.observation_delta_indices],
        "action": [i / fps for i in cfg.action_delta_indices],
    }

    total_episodes = dataset_metadata.total_episodes
    episodes = list(range(total_episodes))
    num_train_episodes = math.floor(total_episodes * 90 / 100)
    train_episodes = episodes[:num_train_episodes]
    val_episodes = list(range(len(episodes[num_train_episodes:])))
    print(f"Number of episodes in full dataset: {total_episodes}")
    print(f"Number of episodes in training dataset (90% subset): {len(train_episodes)}")
    print(f"Number of episodes in validation dataset (10% subset): {len(val_episodes)}")

    train_dataset = LeRobotDataset(
        repo_id,
        root=dataset_root,
        episodes=train_episodes,
        delta_timestamps=delta_timestamps,
    )
    val_dataset = LeRobotDataset(
        repo_id,
        root=dataset_root,
        episodes=val_episodes,
        delta_timestamps=delta_timestamps,
    )
    print(f"Number of frames in training dataset (90% subset): {len(train_dataset)}")
    print(f"Number of frames in validation dataset (10% subset): {len(val_dataset)}")

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=4,
        batch_size=64,
        shuffle=False,
        pin_memory=device != torch.device("cpu"),
        drop_last=False,
    )

    loss_cumsum = 0
    n_examples_evaluated = 0
    for batch in val_dataloader:
        batch = {
            k: (torch.stack(v).to(device, non_blocking=True) if isinstance(v, list) and isinstance(v[0], torch.Tensor)
                else v.to(device, non_blocking=True) if isinstance(v, torch.Tensor)
                else v)
            for k, v in batch.items()
        }
        loss, _ = policy.forward(batch)

        loss_cumsum += loss.item()
        n_examples_evaluated += batch["index"].shape[0]

    average_loss = loss_cumsum / n_examples_evaluated
    print(f"Average loss on validation set: {average_loss:.4f}")


if __name__ == "__main__":
    main()