"""Server benchmark: serial vs Ray-CPU vs Ray-GPU on identical federated workloads.

One workload (N clients x R rounds x (collect + updates), SAC on a MuJoCo
task) is executed in three modes with the SAME federated protocol (FedAvg,
same budget, same evaluation settings):

* ``serial``   — the N client actors run in-process, one after another
  (device=cpu; no serialization, no coordination),
* ``ray-cpu``  — N Ray actors in parallel on CPU,
* ``ray-gpu``  — N Ray actors in parallel, one GPU per actor.

Per-round phase times (broadcast/sync, local training, aggregation) are
measured by driving the actors directly. GPU utilization and memory are
sampled in the background during the ray-gpu mode, and each actor's device
placement is verified via ``get_device_info`` instead of trusting the
``num_gpus`` resource declaration. Startup (actor construction, Ray launch,
numba compilation) is reported separately from steady-state training.

No speedup is presumed: with spare cores and cheap per-update Python
overhead, CPU parallelism may beat GPU parallelism; the data decides.

Run on the GPU server::

    python examples/federated/benchmark_server.py --mode ray-gpu --rounds 100 \
        --clients 4 --out results/benchmark_server.md
"""
from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import threading
import time

import torch

from examples.federated.algorithm_builders import (
    build_sac_mujoco,
    make_mujoco_env_fn,
)
from tianshou.federated.adapter import SACFederatedAdapter
from tianshou.federated.client_actor import TianshouFedClientActor, remote_client_actor
from tianshou.federated.weight_utils import ClientUpdate, LocalBudget, fedavg_aggregate

_MUJOCO_TASKS = {"halfcheetah": "HalfCheetah-v5", "hopper": "Hopper-v5"}
STEADY_START_ROUND = 5  # rounds before this index count as warm-up


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=str, default="halfcheetah", choices=["halfcheetah", "hopper"])
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--collect", type=int, default=1,
                        help="env steps collected before each update; with one env "
                             "per client, collect=1 x updates=1000 gives UTD=1 "
                             "(collect=1000 would mean 1M steps per round!)")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--mode", type=str, required=True,
                        choices=["serial", "ray-cpu", "ray-gpu"])
    parser.add_argument("--repeats", type=int, default=1,
                        help="repeat the whole mode this many times (mean +- std)")
    parser.add_argument("--out", type=str, default="results/benchmark_server.md")
    return parser.parse_args()


class GpuSampler:
    """Samples nvidia-smi utilization/memory in the background.

    Only the cards the experiment actually occupies are sampled, so that
    unrelated co-located load (e.g. keep-alive burners) does not leak into
    the benchmark's utilization numbers.
    """

    def __init__(self, interval: float = 5.0, cards: list[int] | None = None) -> None:
        self.interval = interval
        self.cards = cards
        self.samples: list[tuple[float, float]] = []  # (util %, mem MiB) per card
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        if not self.samples:
            return {"n_samples": 0}
        per_card = list(zip(*self.samples, strict=True))
        return {
            "n_samples": len(self.samples),
            "gpu_util_mean": float(statistics.mean(u for u, _ in per_card[0])),
            "gpu_util_max": float(max(u for u, _ in per_card[0])),
            "gpu_mem_mean_mib": float(statistics.mean(m for _, m in per_card[0])),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                ).stdout.strip().splitlines()
                rows = [
                    (float(row.split(",")[0].strip()), float(row.split(",")[1].strip()))
                    for row in out
                ]
                if self.cards is not None:
                    rows = [rows[i] for i in self.cards]
                self.samples.append(tuple(rows))
            except Exception:  # noqa: BLE001 - sampling is best-effort
                pass
            self._stop.wait(self.interval)


def build_local_clients(args: argparse.Namespace, device: str, seed: int) -> list[TianshouFedClientActor]:
    task = _MUJOCO_TASKS[args.env]
    return [
        TianshouFedClientActor(
            client_id=i,
            env_fn=make_mujoco_env_fn(task),
            algo_builder=lambda: build_sac_mujoco(task=task),
            adapter_factory=SACFederatedAdapter,
            num_train_envs=1,
            buffer_size=200000,
            warmup_steps=2000,
            device=device,
            seed=seed,
        )
        for i in range(args.clients)
    ]


def run_serial(args: argparse.Namespace) -> dict:
    t0 = time.perf_counter()
    clients = build_local_clients(args, device="cpu", seed=42)
    budget = LocalBudget(args.collect, args.updates, args.batch)
    global_weights = clients[0].get_initial_weights()
    per_round: list[float] = []
    client_work = 0.0
    for _ in range(args.rounds):
        tr = time.perf_counter()
        updates: list[ClientUpdate] = []
        for client in clients:
            tw = time.perf_counter()
            updates.append(client.local_train_round(global_weights, budget))
            client_work += time.perf_counter() - tw
        global_weights = fedavg_aggregate(updates)
        per_round.append(time.perf_counter() - tr)
    wall = time.perf_counter() - t0
    return _summarize("serial", wall, per_round, client_work, {})


def run_ray(args: argparse.Namespace, device: str) -> dict:
    import ray

    ray.init(ignore_reinit_error=True)
    try:
        task = _MUJOCO_TASKS[args.env]
        t0 = time.perf_counter()
        if device == "cuda":
            # Fractional GPUs: 4 actors per physical card keeps the card above
            # the cluster's minimum-utilization threshold during CPU-bound
            # phases (one actor alone sits at ~5-7% duty).
            actor_cls = remote_client_actor(num_cpus=8, num_gpus=0.25)
        else:
            actor_cls = remote_client_actor(num_cpus=8)
        clients = [
            actor_cls.remote(
                client_id=i,
                env_fn=make_mujoco_env_fn(task),
                algo_builder=lambda: build_sac_mujoco(task=task),
                adapter_factory=SACFederatedAdapter,
                num_train_envs=1,
                buffer_size=200000,
                warmup_steps=2000,
                device=device,
                seed=42,
            )
            for i in range(args.clients)
        ]
        # Verify device placement per actor instead of trusting num_gpus.
        device_info = ray.get([c.get_device_info.remote() for c in clients])
        device_summary = f"{device} | " + "; ".join(
            f"actor{i}: {d.get('gpu_name') or d['device']} (visible={d['cuda_visible_devices']})"
            for i, d in enumerate(device_info)
        )
        occupied_cards = sorted({
            int(str(d.get("cuda_visible_devices", "")).split(",")[0])
            for d in device_info
            if d.get("device") == "cuda" and str(d.get("cuda_visible_devices", "")).isdigit()
        })
        startup = time.perf_counter() - t0

        budget = LocalBudget(args.collect, args.updates, args.batch)
        global_weights = ray.get(clients[0].get_initial_weights.remote())
        sampler = GpuSampler(cards=occupied_cards or None) if device == "cuda" else None
        if sampler is not None:
            sampler.start()

        per_round: list[float] = []
        phase_sums = {"sync": 0.0, "train": 0.0, "aggregate": 0.0}
        t0 = time.perf_counter()
        for _ in range(args.rounds):
            tr = time.perf_counter()
            weights_ref = ray.put(global_weights)
            ray.get([c.sync_weights.remote(weights_ref) for c in clients])
            t_sync = time.perf_counter()
            updates = ray.get([c.local_train.remote(budget) for c in clients])
            t_train = time.perf_counter()
            global_weights = fedavg_aggregate(updates)
            t_agg = time.perf_counter()
            per_round.append(t_agg - tr)
            phase_sums["sync"] += t_sync - tr
            phase_sums["train"] += t_train - t_sync
            phase_sums["aggregate"] += t_agg - t_train
        wall = time.perf_counter() - t0
        gpu_stats = sampler.stop() if sampler is not None else {}
        client_work = sum(phase_sums.values())
        return _summarize(f"ray-{device}", wall, per_round, client_work,
                          gpu_stats, startup=startup,
                          device_summary=device_summary)
    finally:
        ray.shutdown()


def _summarize(mode: str, wall: float, per_round: list[float], client_work: float,
               gpu_stats: dict, startup: float = 0.0, device_summary: str = "") -> dict:
    steady = per_round[STEADY_START_ROUND:]
    return {
        "mode": mode,
        "wall": wall,
        "startup": startup,
        "steady_round_mean": float(statistics.mean(steady)) if steady else float("nan"),
        "steady_round_std": float(statistics.stdev(steady)) if len(steady) > 1 else 0.0,
        "round_mean_all": float(statistics.mean(per_round)),
        "client_work": client_work,
        "rounds": len(per_round),
        "gpu": gpu_stats,
        "device_summary": device_summary,
    }


def main() -> None:
    args = parse_args()
    # The default 64-thread BLAS dispatch is catastrophic for these small
    # matmuls (measured 4051 ms/update at 64 threads vs 14 ms at 8 on the
    # benchmark container). Pin the thread count so every mode runs in a
    # sane configuration; the comparison stays fair because all three modes
    # share the setting.
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    torch.set_num_threads(8)
    rows: list[dict] = []
    for _ in range(args.repeats):
        if args.mode == "serial":
            rows.append(run_serial(args))
        else:
            device = "cuda" if args.mode == "ray-gpu" else "cpu"
            rows.append(run_ray(args, device))

    if len(rows) > 1:
        walls = [r["wall"] for r in rows]
        print(f"{args.mode} x{len(rows)} repeats: wall = {statistics.mean(walls):.1f} "
              f"+- {statistics.stdev(walls):.1f}s")
    for r in rows:
        print(f"[{r['mode']}] wall={r['wall']:.1f}s startup={r['startup']:.1f}s "
              f"steady_round={r['steady_round_mean']:.2f}+-{r['steady_round_std']:.2f}s "
              f"client_work={r['client_work']:.1f}s")
        if r["gpu"]:
            print(f"  gpu: {r['gpu']}")
        if r["device_summary"]:
            print(f"  devices: {r['device_summary']}")
    with open(args.out, "a") as f:
        for r in rows:
            f.write(
                f"| {r['mode']} | {r['rounds']} | {r['wall']:.1f} | {r['startup']:.1f} | "
                f"{r['steady_round_mean']:.2f} | {r['steady_round_std']:.2f} | "
                f"{r['client_work']:.1f} | {r['gpu'].get('gpu_util_mean', float('nan')):.1f} |\n"
            )


if __name__ == "__main__":
    main()
