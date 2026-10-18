"""Single-process SAC baseline calibration on standard Pendulum-v1.

No Ray, no federation: this isolates the plain tianshou SAC training loop so
the baseline can be trusted before any federated comparison.

Config follows tianshou's own SAC benchmark example
(``examples/mujoco/mujoco_sac.py``): hidden [256, 256], actor/critic lr 1e-3,
gamma 0.99, tau 0.005, fixed alpha 0.2, batch 256, update-to-data ratio 1,
10k random warm-up steps, replay buffer 1e5. Total budget: 100k environment
steps (excluding warm-up), evaluated with 20 deterministic episodes every
5k steps.

Implementation notes (checked against tianshou 2.0.1 source, not modified):

* action scaling: SACPolicy(action_scaling=True) maps tanh outputs to the
  action-space bounds; the actor output is already bounded by tanh.
* evaluation: deterministic (SACPolicy(deterministic_eval=True) uses the
  action-distribution mode outside the training-step context).
* episode termination: the collector computes done = terminated OR truncated
  and stores both flags separately (tianshou/data/collector.py:885).
* warm-up: ``collector.collect(n_step=..., random=True)`` samples actions
  uniformly from the action space (tianshou's benchmark mechanism).

Run::

    python examples/federated/calibrate_sac.py --total-env-steps 100000 \
        --out /tmp/sac_baseline_calibration.csv
"""
from __future__ import annotations

import argparse
import csv

import gymnasium as gym
import numpy as np
import torch

from examples.federated.sac_builder import build_pendulum_sac, make_pendulum_env_fn
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.utils.torch_utils import policy_within_training_step, torch_train_mode

CSV_COLUMNS = [
    "env_steps",
    "grad_steps",
    "mean_return",
    "std_return",
    "actor_loss_mean",
    "critic1_loss_mean",
    "critic2_loss_mean",
    "alpha",
    "buffer_size",
    "utd_ratio",
    "warmup_steps",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-env-steps", type=int, default=100000)
    parser.add_argument("--warmup-steps", type=int, default=10000,
                        help="random-action collection before the first update")
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--collect-chunk", type=int, default=1000,
                        help="env steps collected per loop iteration (updates follow 1:1)")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--buffer", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", nargs="+", type=int, default=[256, 256])
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gravity", type=float, default=10.0,
                        help="pendulum gravity (client heterogeneity); m=l=1 always")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="/tmp/sac_baseline_calibration.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    algo = build_pendulum_sac(
        lr=args.lr,
        hidden_sizes=args.hidden,
        tau=args.tau,
        alpha=args.alpha,
        gravity=args.gravity,
    )
    algo.to(args.device)

    env_fn = make_pendulum_env_fn(args.gravity)
    train_env = DummyVectorEnv([env_fn])
    test_envs = DummyVectorEnv([env_fn for _ in range(10)])
    buffer = VectorReplayBuffer(args.buffer, buffer_num=1)
    train_collector = Collector(algo, train_env, buffer, exploration_noise=False)
    test_collector = Collector(algo, test_envs, exploration_noise=False)
    train_collector.reset()
    test_collector.reset()

    print(
        f"config: lr={args.lr} hidden={args.hidden} tau={args.tau} alpha={args.alpha} "
        f"batch={args.batch} buffer={args.buffer} UTD=1.0 warmup={args.warmup_steps} "
        f"budget={args.total_env_steps} (excl. warmup) gravity={args.gravity} "
        f"seed={args.seed} device={args.device}"
    )

    # Warm-up: fill the buffer with random-action transitions.
    train_collector.collect(n_step=args.warmup_steps, random=True)

    csv_file = open(args.out, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(CSV_COLUMNS)
    print("wrote header to", args.out)

    env_steps = 0
    grad_steps = 0
    loss_sums = {"actor": 0.0, "critic1": 0.0, "critic2": 0.0}
    loss_counts = 0
    next_eval_at = args.eval_every

    while env_steps < args.total_env_steps:
        chunk = min(args.collect_chunk, args.total_env_steps - env_steps)
        with policy_within_training_step(algo.policy):
            train_collector.collect(n_step=chunk)
            env_steps += chunk
            # UTD = 1: one gradient step per collected env step.
            with torch_train_mode(algo):
                for _ in range(chunk):
                    stats = algo.update(sample_size=args.batch, buffer=buffer)
                    loss_sums["actor"] += stats.actor_loss
                    loss_sums["critic1"] += stats.critic1_loss
                    loss_sums["critic2"] += stats.critic2_loss
                    loss_counts += 1
            grad_steps += chunk

        if env_steps >= next_eval_at or env_steps >= args.total_env_steps:
            eval_stats = test_collector.collect(n_episode=args.eval_episodes)
            returns = eval_stats.returns
            row = [
                env_steps,
                grad_steps,
                float(returns.mean()),
                float(returns.std()),
                loss_sums["actor"] / loss_counts,
                loss_sums["critic1"] / loss_counts,
                loss_sums["critic2"] / loss_counts,
                float(algo.alpha.value),
                len(buffer),
                grad_steps / max(env_steps, 1),
                args.warmup_steps,
            ]
            writer.writerow(row)
            csv_file.flush()
            print(
                f"steps={env_steps:6d} | return={row[2]:8.1f} ± {row[3]:6.1f} | "
                f"loss actor={row[4]:6.3f} c1={row[5]:6.3f} c2={row[6]:6.3f} | "
                f"alpha={row[7]:.3f} | buffer={row[8]}"
            )
            loss_sums = {"actor": 0.0, "critic1": 0.0, "critic2": 0.0}
            loss_counts = 0
            next_eval_at += args.eval_every

    csv_file.close()
    print(f"calibration done: {env_steps} env steps, {grad_steps} grad steps, "
          f"CSV at {args.out}")


if __name__ == "__main__":
    main()
