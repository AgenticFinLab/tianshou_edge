"""Serial vs Ray parallel benchmark for identical federated client tasks.

The same workload (3 clients, PPO on CartPole, fixed rounds x steps x
epochs) is executed twice:

* **serial**: the three client actors run in-process, one after another
  (no serialization of weights, no coordination — pure client compute);
* **ray**: the same actors as Ray remote actors, driven by the coordinator.

Reported metrics separate **platform overhead** (Ray communication and
coordination time) from **parallelism gains** (wall-time speedup). No
speedup is presumed: on CPU-bound clients with spare cores the Ray run is
expected to be faster; on a single shared GPU the serialization there can
dominate instead.

Run from the repo root::

    python examples/federated/benchmark_serial_vs_ray.py --rounds 5

Results are appended to ``results/benchmark_serial_vs_ray.md``.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from examples.federated.algorithm_builders import build_ppo_cartpole, make_cartpole_env_fn
from tianshou.federated.adapter import OnPolicyFederatedAdapter
from tianshou.federated.client_actor import TianshouFedClientActor, remote_client_actor
from tianshou.federated.coordinator import FederatedCoordinator
from tianshou.federated.weight_utils import ClientUpdate, LocalBudget, fedavg_aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--collect", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def build_local_clients(args: argparse.Namespace) -> list[TianshouFedClientActor]:
    """Three in-process client actors with identical tasks (same env, same seed)."""
    clients = []
    for i in range(args.clients):
        clients.append(
            TianshouFedClientActor(
                client_id=i,
                env_fn=make_cartpole_env_fn(9.8),
                algo_builder=build_ppo_cartpole,
                adapter_factory=OnPolicyFederatedAdapter,
                num_train_envs=4,
                buffer_size=20000,
                device=args.device,
                seed=42,
            )
        )
    return clients


def run_serial(args: argparse.Namespace) -> tuple[float, float]:
    """Run the workload sequentially in-process; return (wall, client_work)."""
    clients = build_local_clients(args)
    budget = LocalBudget(args.collect, args.epochs, args.batch)
    global_weights = clients[0].get_initial_weights()

    client_work = 0.0
    t0 = time.perf_counter()
    for _ in range(args.rounds):
        updates: list[ClientUpdate] = []
        for client in clients:
            t_work = time.perf_counter()
            updates.append(client.local_train_round(global_weights, budget))
            client_work += time.perf_counter() - t_work
        global_weights = fedavg_aggregate(updates)
    wall = time.perf_counter() - t0
    return wall, client_work


def run_ray(args: argparse.Namespace) -> tuple[float, float]:
    """Run the same workload on Ray; return (wall, actor_build_time)."""
    import ray

    t_build = time.perf_counter()
    actor_cls = remote_client_actor(num_cpus=1)
    clients = [
        actor_cls.remote(
            client_id=i,
            env_fn=make_cartpole_env_fn(9.8),
            algo_builder=build_ppo_cartpole,
            adapter_factory=OnPolicyFederatedAdapter,
            num_train_envs=4,
            buffer_size=20000,
            device=args.device,
            seed=42,
        )
        for i in range(args.clients)
    ]
    coordinator = FederatedCoordinator(clients)  # fetches initial weights
    actor_build = time.perf_counter() - t_build

    budget = LocalBudget(args.collect, args.epochs, args.batch)
    t0 = time.perf_counter()
    coordinator.run(
        num_rounds=args.rounds,
        budget=budget,
        verbose=False,
    )
    wall = time.perf_counter() - t0
    ray.shutdown()
    return wall, actor_build


def main() -> None:
    args = parse_args()
    env_steps_per_round = args.collect * 4  # 4 parallel envs per client
    total_env_steps = env_steps_per_round * args.rounds * args.clients

    serial_wall, client_work = run_serial(args)
    ray_wall, actor_build = run_ray(args)

    speedup = serial_wall / ray_wall
    # Platform overhead: Ray wall minus the ideal parallel wall (the client
    # compute spread evenly over the clients). A negative value means the
    # communication + coordination cost is smaller than the measurement
    # noise, i.e. effectively zero for this payload size.
    ideal_parallel_wall = client_work / args.clients
    comm_overhead = ray_wall - ideal_parallel_wall
    comm_overhead_per_round = comm_overhead / args.rounds

    print(f"workload: {args.clients} clients x {args.rounds} rounds x "
          f"({args.collect} steps x 4 envs + {args.epochs} PPO epochs) | "
          f"total env steps: {total_env_steps}")
    print(f"serial : wall={serial_wall:.1f}s | client compute={client_work:.1f}s | "
          f"throughput={total_env_steps / serial_wall:.0f} steps/s")
    print(f"ray    : wall={ray_wall:.1f}s (actor build {actor_build:.1f}s) | "
          f"throughput={total_env_steps / ray_wall:.0f} steps/s")
    print(f"speedup: {speedup:.2f}x")
    print(f"platform overhead (comm + coordination): {comm_overhead:.1f}s total, "
          f"{comm_overhead_per_round:.2f}s per round")

    with open("results/benchmark_serial_vs_ray.md", "a") as f:
        f.write(
            f"| {args.clients}x{args.rounds}x({args.collect}x4+{args.epochs}) "
            f"| {serial_wall:.1f} | {client_work:.1f} | {ray_wall:.1f} "
            f"| {actor_build:.1f} | {speedup:.2f} | {comm_overhead_per_round:.2f} "
            f"| {total_env_steps / serial_wall:.0f} / {total_env_steps / ray_wall:.0f} |\n"
        )


if __name__ == "__main__":
    main()
