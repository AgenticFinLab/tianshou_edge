"""SAC federated comparison on gravity-heterogeneous Pendulum (step 6 protocol).

Config matches the validated single-process baseline (see
``examples/federated/calibrate_sac.py`` and
``examples/federated/sac_builder.py``): hidden [256, 256], lr 1e-3,
tau 0.005, fixed alpha 0.2, batch 256, update-to-data ratio 1, 10k random
warm-up steps per client. Unified budget: 60k environment steps per client
(15 rounds x 4000 collected steps + 4000 gradient steps; the 4 parallel
training envs make collect(n_step=1) yield 4 env steps, so the adapter runs
4000 collect+update iterations per round at UTD = 1).

Methods (same budget, seeds, envs, and evaluation protocol everywhere):

* ``local_only`` — no federation (identity aggregation),
* ``fedavg_all`` — FedAvg over actor + both critics + target critics,
* ``fedavg_actor_only`` — FedAvg over the actor only; critics, target
  critics and alpha stay client-private,
* ``fedprox_actor`` — actor-only FedAvg plus a proximal term (mu) on the
  actor only (value networks are never pulled towards old global weights
  because their TD targets are non-stationary).

Clients: gravity g in {8, 10, 12} (m = l = 1). Evaluation returns are NOT
comparable across clients (different physics); methods are compared per
client. Final per-client scores are means over the last 6 evaluation rounds
(20 deterministic episodes each), then aggregated over seeds as mean +/- std
with seeds as the statistical unit.

Run from the repo root (one background job per seed for parallelism)::

    python examples/federated/run_sac_experiment.py --seeds 42 --out /tmp/sac_fed_s42.csv
    python examples/federated/run_sac_experiment.py --seeds 43 --out /tmp/sac_fed_s43.csv
    python examples/federated/run_sac_experiment.py --seeds 44 --out /tmp/sac_fed_s44.csv
"""
from __future__ import annotations

import argparse
import csv
import statistics

from examples.federated.sac_builder import build_pendulum_sac, make_pendulum_env_fn
from tianshou.federated.adapter import SACFederatedAdapter
from tianshou.federated.client_actor import remote_client_actor
from tianshou.federated.coordinator import FederatedCoordinator
from tianshou.federated.weight_utils import LocalBudget

PENDULUM_GRAVITIES = [8.0, 10.0, 12.0]
METHODS = ["local_only", "fedavg_all", "fedavg_actor_only", "fedprox_actor"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=15,
                        help="rounds; budget = rounds x 4000 env steps per client "
                             "(collect=1 x updates=4000 per round, UTD = 1; note the "
                             "vector env steps 4 envs per collect(n_step=1) call)")
    parser.add_argument("--envs-per-client", type=int, default=4)
    parser.add_argument("--collect", type=int, default=1,
                        help="env steps requested per local update; with 4 parallel "
                             "envs one collect(n_step=1) yields 4 env steps, so "
                             "collect=1 x updates=4000 gives UTD = 1")
    parser.add_argument("--updates", type=int, default=4000,
                        help="local gradient steps per client per round (UTD = 1)")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--buffer", type=int, default=100000)
    parser.add_argument("--warmup-steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mu", type=float, default=0.05,
                        help="FedProx proximal term strength (fedprox_actor only)")
    parser.add_argument("--eval-every", type=int, default=1,
                        help="evaluation every N rounds (= N x 4000 env steps)")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-cpus", type=float, default=1)
    parser.add_argument("--out", type=str, default=None,
                        help="CSV file for per-round per-client eval returns")
    return parser.parse_args()


def method_config(method: str, args: argparse.Namespace) -> tuple[str, str, float]:
    """Return (aggregation mode, agg_scope, fedprox mu) for a method."""
    if method == "local_only":
        return "local_only", "all", 0.0
    if method == "fedavg_all":
        return "fedavg", "all", 0.0
    if method == "fedavg_actor_only":
        return "fedavg", "actor_only", 0.0
    if method == "fedprox_actor":
        return "fedavg", "actor_only", args.mu
    raise ValueError(f"unknown method {method!r}")


def build_clients(args: argparse.Namespace, agg_scope: str, mu: float, seed: int) -> list:
    actor_cls = remote_client_actor(num_cpus=args.num_cpus)
    return [
        actor_cls.remote(
            client_id=i,
            env_fn=make_pendulum_env_fn(gravity),
            algo_builder=lambda: build_pendulum_sac(lr=args.lr),
            adapter_factory=lambda algo: SACFederatedAdapter(algo, agg_scope=agg_scope),
            num_train_envs=args.envs_per_client,
            buffer_size=args.buffer,
            warmup_steps=args.warmup_steps,
            device=args.device,
            fedprox_mu=mu,
            seed=seed,
        )
        for i, gravity in enumerate(PENDULUM_GRAVITIES)
    ]


def final_per_client(history: list[dict], last_n: int = 6) -> dict[int, float]:
    """Per-client mean over the last ``last_n`` evaluation rounds."""
    per_client: dict[int, list[float]] = {}
    for summary in history:
        for client_id, mean_return in summary.get("eval_returns", {}).items():
            per_client.setdefault(client_id, []).append(mean_return)
    return {
        client_id: statistics.mean(returns[-last_n:])
        for client_id, returns in per_client.items()
        if returns
    }


def run_method(
    args: argparse.Namespace,
    method: str,
    seed: int,
    on_round=None,
) -> list[dict]:
    aggregation, agg_scope, mu = method_config(method, args)
    clients = build_clients(args, agg_scope, mu, seed)
    coordinator = FederatedCoordinator(clients)
    return coordinator.run(
        num_rounds=args.rounds,
        budget=LocalBudget(
            num_env_steps_per_update=args.collect,
            num_updates=args.updates,
            batch_size=args.batch,
        ),
        aggregation=aggregation,
        eval_every=args.eval_every,
        eval_n_episode=args.eval_episodes,
        seed=seed,
        on_round=on_round,
    )


def main() -> None:
    args = parse_args()
    import ray  # lazy import so --help works without ray

    ray.init(ignore_reinit_error=True)
    try:
        print(
            f"gravities={PENDULUM_GRAVITIES} methods={args.methods} seeds={args.seeds} "
            f"budget={args.rounds}x({args.collect} steps + {args.updates} updates)/client "
            f"warmup={args.warmup_steps} eval={args.eval_episodes} episodes/"
            f"{args.eval_every} rounds device={args.device}"
        )
        # Stream evaluation rows to disk as they are produced, so that an
        # interrupted run loses at most the current round.
        csv_file = open(args.out, "w", newline="") if args.out else None
        writer = csv.writer(csv_file) if csv_file else None
        if writer is not None:
            writer.writerow(["method", "seed", "round", "client_id", "eval_return"])

        per_seed_scores: dict[tuple[str, int], dict[int, float]] = {}
        for seed in args.seeds:
            for method in args.methods:
                print(f"\n===== seed {seed} | method: {method} =====")

                def make_on_round(method: str, seed: int):
                    def on_round(summary: dict) -> None:
                        for client_id, mean_return in summary.get("eval_returns", {}).items():
                            if writer is not None:
                                writer.writerow([method, seed, summary["round"],
                                                 client_id, mean_return])
                                csv_file.flush()

                    return on_round

                history = run_method(args, method, seed, on_round=make_on_round(method, seed))
                per_seed_scores[(method, seed)] = final_per_client(history)

        print("\n===== final per-client eval return (last 6 eval rounds, mean over seeds) =====")
        print("returns are NOT comparable across clients (different physics)")
        header = " " * 18 + "".join(f"client{i}(g={g:4.1f})" for i, g in enumerate(
            PENDULUM_GRAVITIES))
        print(header)
        for method in args.methods:
            cells = []
            for client_id in range(len(PENDULUM_GRAVITIES)):
                scores = [
                    per_seed_scores[(method, seed)][client_id] for seed in args.seeds
                ]
                mean = statistics.mean(scores)
                std = statistics.stdev(scores) if len(scores) > 1 else 0.0
                cells.append(f"{mean:>18.1f}±{std:<5.1f}")
            print(f"{method:18s}" + "".join(cells))

        if csv_file is not None:
            csv_file.close()
            print(f"\nper-round per-client eval returns written to {args.out}")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
