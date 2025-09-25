import argparse
import os
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import create_dataset
from model import DiffusionModule
from scheduler import DDPMScheduler


def train_one_step(ddpm, optimizer, device, batch):
    optimizer.zero_grad()
    x, y = batch
    x, y = x.to(device), y.to(device)

    loss = ddpm.compute_loss(x, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def main(args):
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # dataset + dataloader
    dataset = create_dataset(args.dataset_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # scheduler + model
    var_scheduler = DDPMScheduler(
        num_train_timesteps=args.total_steps,
        beta_1=args.beta_1,
        beta_T=args.beta_T,
        mode=args.mode,
    ).to(device)

    ddpm = DiffusionModule(network=None, var_scheduler=var_scheduler).to(device)
    optimizer = optim.Adam(ddpm.parameters(), lr=args.lr)

    # checkpoint dir
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 記錄時間
    start_time = time.time()

    step = 0
    for epoch in range(args.epochs):
        for batch in dataloader:
            iter_start = time.time()

            loss = train_one_step(ddpm, optimizer, device, batch)

            iter_time = time.time() - iter_start
            step += 1

            # log
            if step % args.log_interval == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / step
                remaining_steps = args.total_steps - step
                eta = remaining_steps * avg_time / 60  # 分鐘

                print(
                    f"[Step {step}/{args.total_steps}] "
                    f"loss={loss:.4f} | avg_time={avg_time:.3f}s/step | ETA={eta:.1f} min"
                )

            # checkpoint
            if step % args.ckpt_interval == 0:
                ckpt_path = ckpt_dir / f"ckpt_{step}.pt"
                torch.save({
                    "step": step,
                    "model": ddpm.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": var_scheduler.state_dict(),
                }, ckpt_path)
                print(f"Checkpoint saved at {ckpt_path}")

            if step >= args.total_steps:
                print("Training finished!")
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default="data/afhq/train")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--ckpt_interval", type=int, default=1000)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--mode", type=str, default="linear",
                        choices=["linear", "cosine", "quad"])
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)

    args = parser.parse_args()
    main(args)
