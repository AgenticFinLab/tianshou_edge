"""Compute-scaling sweep: ms per SAC update across model size, batch size, and device.

Finds the crossover where GPU updates become cheaper than CPU updates, so
that GPU-hungry platform benchmarks can be configured at the crossover point.

Grid: hidden_sizes in {[256,256], [512,512], [1024,1024]} x
      batch_size in {256, 1024, 4096, 16384} x device in {cpu, cuda}.

CPU runs pin OMP_NUM_THREADS=8 (the sane configuration measured earlier; the
64-thread default is ~290x slower). Run with nothing else sharing the CPU
for clean numbers.

Results are appended to results/update_cost_sweep.md.
"""
from __future__ import annotations

import os
import time

import gymnasium
import torch

os.environ.setdefault("OMP_NUM_THREADS", "8")
torch.set_num_threads(8)

from examples.federated.algorithm_builders import build_sac_mujoco
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.utils.torch_utils import policy_within_training_step, torch_train_mode

HIDDENS = [[256, 256], [512, 512], [1024, 1024]]
BATCHES = [256, 1024, 4096, 16384]
WARMUP_UPDATES = 3
TIMED_UPDATES = 20


def fill_buffer(algo, needed: int) -> VectorReplayBuffer:
    """Fill a buffer with random transitions; reuse one env per config."""
    buf = VectorReplayBuffer(needed + 2000, 1)
    col = Collector(algo, DummyVectorEnv([lambda: gymnasium.make("HalfCheetah-v5")]), buf)
    col.reset()
    col.collect(n_step=needed + 1000, random=True)
    return buf


def measure(hidden: list[int], batch: int, device: str) -> float:
    algo = build_sac_mujoco(task="HalfCheetah-v5", hidden_sizes=hidden).to(device)
    buf = fill_buffer(algo, max(batch, 2000))
    with policy_within_training_step(algo.policy), torch_train_mode(algo):
        for _ in range(WARMUP_UPDATES):
            algo.update(sample_size=batch, buffer=buf)
        t0 = time.perf_counter()
        for _ in range(TIMED_UPDATES):
            algo.update(sample_size=batch, buffer=buf)
        dt = time.perf_counter() - t0
    return dt / TIMED_UPDATES * 1000.0


def main() -> None:
    rows: list[tuple[int, int, float, float]] = []  # (batch, ms_cpu, ms_cuda)
    for hidden in HIDDENS:
        for batch in BATCHES:
            ms_cpu = measure(hidden, batch, "cpu")
            ms_cuda = measure(hidden, batch, "cuda")
            rows.append((batch, ms_cpu, ms_cuda))
            print(f"hidden={hidden} batch={batch:5d}: cpu={ms_cpu:8.1f} ms | "
                  f"cuda={ms_cuda:8.1f} ms | ratio={ms_cpu/ms_cuda:6.2f}x", flush=True)
        with open("results/update_cost_sweep.md", "a") as f:
            f.write(f"\n## hidden_sizes = {hidden} (ms/update, OMP=8)\n")
            f.write("| batch | cpu | cuda | cpu/cuda |\n|---|---|---|---|\n")
            for batch, ms_cpu, ms_cuda in rows:
                f.write(f"| {batch} | {ms_cpu:.1f} | {ms_cuda:.1f} | {ms_cpu/ms_cuda:.2f} |\n")
            rows.clear()


if __name__ == "__main__":
    main()
