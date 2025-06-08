#!/usr/bin/env python3
"""Evaluate a trained IQL agent on a dataset."""

import argparse
import json
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

# ---- Dataset and utilities copied from train.py ----


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, v, n=1):
        self.val = v
        self.sum += v * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0


def _discover_episode_files(root: Path):
    return sorted(root.glob("**/episode_*.parquet"))


class LerobotDataset(Dataset):
    """Streams transitions from LeRobot parquet episodes."""

    def __init__(
        self, root: str | Path, image_size: int = 128, qvel_dim: int = 3, text_model: str = "all-MiniLM-L6-v2"
    ):
        self.root = Path(root)
        self.files = _discover_episode_files(self.root)
        assert self.files, f"No episode parquet files found in {root}"

        self.file_meta = []
        self.index = []
        for fi, fp in enumerate(self.files):
            n_rows = pq.ParquetFile(fp).metadata.num_rows
            self.file_meta.append((fp, n_rows))
            self.index.extend([(fi, ri) for ri in range(n_rows)])

        self.task_to_prompt: Dict[int, str] = {}
        tasks_jsonl = self.root / "meta" / "tasks.jsonl"
        if tasks_jsonl.exists():
            with open(tasks_jsonl) as f:
                for line in f:
                    item = json.loads(line)
                    self.task_to_prompt[int(item["task_index"])] = item["task"].strip()
        else:
            tasks_dir = self.root / "tasks"
            task_txts = sorted(tasks_dir.glob("*.txt"))
            for p in task_txts:
                try:
                    idx = int(p.stem.split("_")[-1]) if "_" in p.stem else int(p.stem)
                except ValueError:
                    continue
                self.task_to_prompt[idx] = p.read_text().strip()
        assert self.task_to_prompt, "No task prompt files found."

        self.text_encoder = SentenceTransformer(text_model, device="cpu")
        self.prompt_cache = {}
        for idx, prompt in self.task_to_prompt.items():
            emb = self.text_encoder.encode(prompt, convert_to_numpy=True)
            self.prompt_cache[idx] = torch.from_numpy(emb.astype(np.float32))

        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.img_tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((360, 640)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self._curr_file_idx = None
        self._curr_df = None
        self.qvel_dim = qvel_dim

    def _load_file(self, file_idx: int):
        path = self.file_meta[file_idx][0]
        if self._curr_df is not None:
            del self._curr_df
        self._curr_df = pq.read_table(path).to_pandas()
        self._curr_file_idx = file_idx

    def _get_row(self, file_idx: int, row_idx: int):
        if file_idx != self._curr_file_idx or self._curr_df is None:
            self._load_file(file_idx)
        return self._curr_df.iloc[row_idx]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, global_idx):
        file_idx, row_idx = self.index[global_idx]
        row = self._get_row(file_idx, row_idx)

        img_bytes = row["observation.images.perspective"]
        img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img = self.img_tf(img_rgb)

        qvel = torch.tensor(np.array(row["observation.state"][: self.qvel_dim], dtype=np.float32, copy=True))
        act = torch.tensor(np.array(row["action"], dtype=np.float32, copy=True))
        rew = torch.as_tensor(row["next.reward"][0], dtype=torch.float32)
        done_flag = bool(row["next.done"][0])
        done = torch.as_tensor(float(done_flag))

        task_idx = int(row["task_index"][0]) if "task_index" in row else 0
        prompt_emb = self.prompt_cache[task_idx]

        if done_flag or row_idx + 1 >= self.file_meta[file_idx][1]:
            nxt_img = torch.zeros_like(img)
            nxt_qvel = torch.zeros_like(qvel)
        else:
            nxt_row = self._get_row(file_idx, row_idx + 1)
            nxt_b = nxt_row["observation.images.perspective"]
            nxt_np = cv2.imdecode(np.frombuffer(nxt_b, np.uint8), cv2.IMREAD_COLOR)
            nxt_rgb = cv2.cvtColor(nxt_np, cv2.COLOR_BGR2RGB)
            nxt_img = self.img_tf(nxt_rgb)
            nxt_qvel = torch.tensor(
                np.array(nxt_row["observation.state"][: self.qvel_dim], dtype=np.float32, copy=True)
            )

        return img, qvel, prompt_emb, act, rew, done, nxt_img, nxt_qvel


# ---- Networks copied from train.py ----


class Policy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs):
        h = self.net(obs)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), -5, 2)
        std = torch.exp(log_std)
        return mu, std


class QFunction(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


class ValueFunction(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class IQLAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: int = 1024,
        expectile: float = 0.7,
        temperature: float = 3.0,
        gamma: float = 0.99,
        lr: float = 3e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.expectile = expectile
        self.temperature = temperature
        self.gamma = gamma

        self.pi = Policy(obs_dim, act_dim, hidden).to(device)
        self.q1 = QFunction(obs_dim, act_dim, hidden).to(device)
        self.q2 = QFunction(obs_dim, act_dim, hidden).to(device)
        self.v = ValueFunction(obs_dim, hidden).to(device)

    def _expectile_loss(self, diff):
        w = torch.where(diff > 0, self.expectile, 1 - self.expectile)
        return (w * diff.pow(2)).mean()

    def compute_losses(self, batch):
        obs, act, rew, next_obs, done = [b.to(self.device) for b in batch]
        with torch.no_grad():
            target_v = self.v(next_obs)
            target_q = rew + self.gamma * (1 - done) * target_v

        q1_pred, q2_pred = self.q1(obs, act), self.q2(obs, act)
        q_loss = (q1_pred - target_q).pow(2).mean() + (q2_pred - target_q).pow(2).mean()

        q_min = torch.minimum(q1_pred, q2_pred)
        v_pred = self.v(obs)
        v_loss = self._expectile_loss(q_min - v_pred)

        adv = q_min - v_pred
        weights = torch.exp(adv / self.temperature).clamp(max=10.0)
        mu, std = self.pi(obs)
        dist = torch.distributions.Normal(mu, std)
        log_prob = dist.log_prob(act).sum(-1)
        pi_loss = -(weights * log_prob).mean()

        return {"q": q_loss.item(), "v": v_loss.item(), "pi": pi_loss.item()}


# ---- Argument parsing ----


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained IQL agent")
    parser.add_argument("--data-root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--checkpoint", type=str, required=True, help="Trained model checkpoint")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--qvel-dim", type=int, default=3)
    parser.add_argument("--text-model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--expectile", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


# ---- Evaluation ----


def evaluate(cfg):
    print(f"[INFO] Using device: {cfg.device}")

    ds = LerobotDataset(
        cfg.data_root, image_size=cfg.image_size, qvel_dim=cfg.qvel_dim, text_model=cfg.text_model
    )
    dl = DataLoader(ds, batch_size=cfg.batch, shuffle=False, num_workers=cfg.workers, pin_memory=True)

    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    backbone.fc = nn.Identity()
    backbone.eval()
    backbone.to(cfg.device)
    for p in backbone.parameters():
        p.requires_grad = False

    feat_dim = backbone(torch.zeros(1, 3, 360, 640, device=cfg.device)).shape[1]
    prompt_dim = next(iter(ds.prompt_cache.values())).numel()
    obs_dim = feat_dim + cfg.qvel_dim + prompt_dim
    act_dim = ds[0][3].numel()

    agent = IQLAgent(
        obs_dim,
        act_dim,
        hidden=cfg.hidden,
        expectile=cfg.expectile,
        temperature=cfg.temperature,
        gamma=cfg.gamma,
        device=cfg.device,
    )

    checkpoint = torch.load(cfg.checkpoint, map_location=cfg.device)
    agent.pi.load_state_dict(checkpoint["policy"])
    agent.q1.load_state_dict(checkpoint["q1"])
    agent.q2.load_state_dict(checkpoint["q2"])
    agent.v.load_state_dict(checkpoint["v"])

    meters = {k: AverageMeter() for k in ("q", "v", "pi")}

    with torch.no_grad():
        for batch in tqdm(dl, desc="Eval"):
            (img, qvel, prompt, act, rew, done, nxt_img, nxt_qvel) = [
                b.to(cfg.device, non_blocking=True) for b in batch
            ]
            feat = backbone(img)
            nfeat = backbone(nxt_img)
            obs = torch.cat([feat, qvel, prompt], dim=1)
            nobs = torch.cat([nfeat, nxt_qvel, prompt], dim=1)
            losses = agent.compute_losses([obs, act, rew, nobs, done])
            for k, v in losses.items():
                meters[k].update(v, n=img.size(0))

    # Compute dataset episode returns
    ep_returns = []
    ep_ret = 0.0
    for i in range(len(ds)):
        _, _, _, _, rew, done, _, _ = ds[i]
        ep_ret += float(rew)
        if done.item():
            ep_returns.append(ep_ret)
            ep_ret = 0.0
    if ep_ret > 0:
        ep_returns.append(ep_ret)

    avg_ret = float(np.mean(ep_returns)) if ep_returns else 0.0

    print(f"Q loss: {meters['q'].avg:.4f}")
    print(f"V loss: {meters['v'].avg:.4f}")
    print(f"Policy loss: {meters['pi'].avg:.4f}")
    print(f"Average episode return: {avg_ret:.2f} over {len(ep_returns)} episodes")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
