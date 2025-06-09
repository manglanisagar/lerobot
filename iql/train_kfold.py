#!/usr/bin/env python3
"""K-fold cross-validation for the IQL agent."""

import argparse
from typing import Dict, List
from pathlib import Path

import numpy as np
import torch
import json
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from torchvision import models
from tqdm import tqdm

from iql.train import AverageMeter, IQLAgent, LerobotDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train an IQL agent with k-fold cross-validation")
    parser.add_argument("--data-root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per fold")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--qvel-dim", type=int, default=3)
    parser.add_argument("--text-model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--expectile", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="runs/iql_kfold", help="Directory to save checkpoints")
    return parser.parse_args()


def compute_episode_metrics(ds: LerobotDataset, indices: List[int]) -> Dict[str, float]:
    """Compute return-based metrics from the provided dataset subset."""
    ep_returns: List[float] = []
    ep_lens: List[int] = []
    ep_ret = 0.0
    ep_len = 0
    for idx in sorted(indices):
        _, _, _, _, rew, done, _, _ = ds[idx]
        ep_ret += float(rew)
        ep_len += 1
        if float(done):
            ep_returns.append(ep_ret)
            ep_lens.append(ep_len)
            ep_ret = 0.0
            ep_len = 0
    if ep_len > 0:
        ep_returns.append(ep_ret)
        ep_lens.append(ep_len)

    if ep_returns:
        avg_return = float(np.mean(ep_returns))
        ret_std = float(np.std(ep_returns))
        max_ret = float(np.max(ep_returns))
        min_ret = float(np.min(ep_returns))
        success_rate = float(np.mean([r > 5 for r in ep_returns]))
        mean_len = float(np.mean(ep_lens))
    else:
        avg_return = ret_std = max_ret = min_ret = success_rate = mean_len = 0.0

    return {
        "avg_return": avg_return,
        "return_std": ret_std,
        "max_return": max_ret,
        "min_return": min_ret,
        "success_rate": success_rate,
        "mean_episode_length": mean_len,
    }



def train_fold(cfg, fold, train_idx, val_idx, ds, backbone, obs_dim, act_dim):
    print(f"\n[INFO] Starting Fold {fold}/{cfg.folds}")
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, sorted(val_idx))

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=4,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
        persistent_workers=False,
    )

    agent = IQLAgent(
        obs_dim,
        act_dim,
        hidden=cfg.hidden,
        expectile=cfg.expectile,
        temperature=cfg.temperature,
        gamma=cfg.gamma,
        lr=cfg.lr,
        device=cfg.device,
    )

    for ep in range(1, cfg.epochs + 1):
        meters = {k: AverageMeter() for k in ("q", "v", "pi")}
        for batch in tqdm(train_dl, desc=f"Fold {fold} Epoch {ep} [train]"):
            img, qvel, prompt, act, rew, done, nxt_img, nxt_qvel = [
                b.to(cfg.device, non_blocking=True) for b in batch
            ]
            with torch.no_grad():
                feat = backbone(img)
                nfeat = backbone(nxt_img)
            obs = torch.cat([feat, qvel, prompt], dim=1)
            nobs = torch.cat([nfeat, nxt_qvel, prompt], dim=1)
            stats = agent.update([obs, act, rew, nobs, done])
            for k, v in stats.items():
                meters[k].update(v, n=img.size(0))
        msg = " ".join(f"{k}:{m.avg:.4f}" for k, m in meters.items())
        print(f"Fold {fold} Epoch {ep} | {msg}")

    # Validation losses
    val_m = {k: AverageMeter() for k in ("q", "v", "pi")}
    with torch.no_grad():
        for batch in tqdm(val_dl, desc=f"Fold {fold} [val]"):
            img, qvel, prompt, act, rew, done, nxt_img, nxt_qvel = [
                b.to(cfg.device, non_blocking=True) for b in batch
            ]
            feat = backbone(img)
            nfeat = backbone(nxt_img)
            obs = torch.cat([feat, qvel, prompt], dim=1)
            nobs = torch.cat([nfeat, nxt_qvel, prompt], dim=1)
            losses = agent.compute_losses([obs, act, rew, nobs, done], cfg.epochs)
            for k, v in losses.items():
                val_m[k].update(v, n=img.size(0))

    # Compute metrics
    metrics = {
        "q_loss": val_m["q"].avg,
        "v_loss": val_m["v"].avg,
        "pi_loss": val_m["pi"].avg,
    }
    metrics.update(compute_episode_metrics(ds, sorted(val_idx)))

    # Save checkpoint
    ckpt_dir = Path(cfg.out_dir) / f"fold_{fold}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "final_model.pt"
    torch.save({
        "policy": agent.pi.state_dict(),
        "q1": agent.q1.state_dict(),
        "q2": agent.q2.state_dict(),
        "v": agent.v.state_dict(),
        "cfg": vars(cfg),
        "fold": fold,
    }, ckpt_path)
    print(f"[INFO] Saved checkpoint: {ckpt_path}")

    metrics_path = ckpt_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Print fold results
    print(f"Fold {fold} results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return metrics


def main(cfg):
    print(f"[INFO] Using device: {cfg.device}")
    ds = LerobotDataset(
        cfg.data_root,
        image_size=cfg.image_size,
        qvel_dim=cfg.qvel_dim,
        text_model=cfg.text_model,
    )

    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    backbone.fc = torch.nn.Identity()
    backbone.eval()
    backbone.to(cfg.device)
    for p in backbone.parameters():
        p.requires_grad = False

    feat_dim = backbone(torch.zeros(1, 3, 360, 640, device=cfg.device)).shape[1]
    prompt_dim = next(iter(ds.prompt_cache.values())).numel()
    obs_dim = feat_dim + cfg.qvel_dim + prompt_dim
    act_dim = ds[0][3].numel()

    indices = np.arange(len(ds))
    kf = KFold(n_splits=cfg.folds, shuffle=True, random_state=42)

    all_metrics: List[Dict[str, float]] = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices), 1):
        metrics = train_fold(cfg, fold, train_idx.tolist(), val_idx.tolist(), ds, backbone, obs_dim, act_dim)
        all_metrics.append(metrics)

    # Aggregate metrics
    print("\nCross-fold results:")
    keys = all_metrics[0].keys()
    for k in keys:
        vals = [m[k] for m in all_metrics]
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        print(f"{k}: {mean:.4f} ± {std:.4f}")
    
    summary_path = Path(cfg.out_dir) / "kfold_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    main(args)
