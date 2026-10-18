"""Unified config-driven entry for federated DQN / SAC / PPO experiments.

One command runs a complete multi-client federated experiment from a single
YAML (or JSON) config file; see :mod:`tianshou.federated.config` for the
schema and ``examples/federated/config_examples/`` for ready-made configs.

Usage from the repo root (requires the ``ray`` package)::

    python examples/federated/run_federated.py --config examples/federated/config_examples/dqn_cartpole.yaml

Per-round per-client evaluation returns are streamed to the CSV named in
``output.csv_path``; the config itself is echoed to stdout so every run is
reproducible from its log.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

from examples.federated.algorithm_builders import get_algo_builder, get_env_fn
from tianshou.federated.adapter import (
    OffPolicyFederatedAdapter,
    OnPolicyFederatedAdapter,
    SACFederatedAdapter,
)
from tianshou.federated.client_actor import remote_client_actor
from tianshou.federated.config import (
    FederatedConfig,
    load_config,
    per_client_values,
)
from tianshou.federated.coordinator import FederatedCoordinator
from tianshou.federated.weight_utils import LocalBudget


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True,
                        help="path to the experiment config (.yaml or .json)")
    parser.add_argument("--resume", type=str, default=None,
                        help="checkpoint directory to resume from")
    parser.add_argument("--resume-mode", type=str, default="full",
                        choices=["full", "model_only"],
                        help="full: restore weights+optimizers+buffer+RNG; "
                             "model_only: restore aggregated weights only")
    return parser.parse_args()


def adapter_factory(algorithm: str, agg_scope: str) -> callable:
    """Return the adapter factory for the configured algorithm and scope.

    ``agg_scope`` applies to SAC and PPO (actor/critic separation); DQN
    shares its online Q network only and ignores the scope.
    """
    if algorithm == "dqn":
        return OffPolicyFederatedAdapter
    if algorithm == "sac":
        return lambda algo: SACFederatedAdapter(algo, agg_scope=agg_scope)
    if algorithm == "ppo":
        return lambda algo: OnPolicyFederatedAdapter(algo, agg_scope=agg_scope)
    raise ValueError(f"unknown algorithm {algorithm!r}")


def build_clients(cfg: FederatedConfig, seed: int) -> list:
    """Build one client actor per configured client with its own physics."""
    gravities = per_client_values(cfg.heterogeneity.gravity, cfg.clients_count)
    masses = per_client_values(cfg.heterogeneity.mass, cfg.clients_count)
    lengths = per_client_values(cfg.heterogeneity.length, cfg.clients_count)
    torques = (
        per_client_values(cfg.heterogeneity.max_torque, cfg.clients_count)
        if cfg.heterogeneity.max_torque is not None
        else [None] * cfg.clients_count
    )
    actor_cls = remote_client_actor(num_cpus=cfg.ray.num_cpus, num_gpus=cfg.ray.num_gpus)
    clients = []
    for i in range(cfg.clients_count):
        env_fn = get_env_fn(
            cfg.env,
            {
                "gravity": gravities[i],
                "mass": masses[i],
                "length": lengths[i],
                "max_torque": torques[i],
            },
        )
        clients.append(
            actor_cls.remote(
                client_id=i,
                env_fn=env_fn,
                algo_builder=get_algo_builder(cfg.algorithm, cfg.algorithm_config),
                adapter_factory=adapter_factory(cfg.algorithm, cfg.agg_scope),
                num_train_envs=cfg.envs_per_client,
                buffer_size=cfg.buffer_size,
                warmup_steps=cfg.warmup_steps,
                device=cfg.device,
                fedprox_mu=cfg.fedprox_mu,
                seed=seed,
            )
        )
    return clients


def final_scores(history: list[dict], last_n: int = 6) -> dict[int, float]:
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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    print("config:", Path(args.config).resolve())
    print(f"  {cfg.algorithm} on {cfg.env}, seeds={cfg.seeds}, clients={cfg.clients_count}, "
          f"aggregation={cfg.aggregation}, agg_scope={cfg.agg_scope}, "
          f"fedprox_mu={cfg.fedprox_mu}, device={cfg.device}")
    print(f"  budget: {cfg.budget.rounds} rounds x ({cfg.budget.collect_per_update} steps "
          f"+ {cfg.budget.updates_per_round} updates/epochs), batch={cfg.budget.batch_size}")

    import ray  # lazy import so --help works without ray

    ray.init(ignore_reinit_error=True)
    try:
        csv_path = Path(cfg.output.csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.resume else "w"
        csv_file = open(csv_path, mode, newline="")
        writer = csv.writer(csv_file)
        if mode == "w":
            writer.writerow(["algorithm", "seed", "round", "client_id", "eval_return",
                             "aggregation", "agg_scope"])
        history_path = csv_path.with_name(csv_path.stem + "_history.jsonl")
        history_file = open(history_path, "a")

        all_scores: dict[int, dict[int, float]] = {}
        for seed in cfg.seeds:
            print(f"\n===== seed {seed} =====")
            clients = build_clients(cfg, seed)
            coordinator = FederatedCoordinator(clients)

            def on_round(summary: dict, seed: int = seed) -> None:
                for client_id, mean_return in summary.get("eval_returns", {}).items():
                    writer.writerow([cfg.algorithm, seed, summary["round"], client_id,
                                     mean_return, cfg.aggregation, cfg.agg_scope])
                    csv_file.flush()
                history_file.write(json.dumps({"seed": seed, **summary}) + "\n")
                history_file.flush()

            history = coordinator.run(
                num_rounds=cfg.budget.rounds,
                budget=LocalBudget(
                    num_env_steps_per_update=cfg.budget.collect_per_update,
                    num_updates=cfg.budget.updates_per_round,
                    batch_size=cfg.budget.batch_size,
                ),
                aggregation=cfg.aggregation,
                eval_every=cfg.evaluation.every_rounds,
                eval_n_episode=cfg.evaluation.episodes,
                seed=seed,
                on_round=on_round,
                eval_pre_post=cfg.evaluation.pre_post_eval,
                checkpoint_dir=cfg.output.checkpoint_dir,
                resume_from=args.resume,
                resume_mode=args.resume_mode,
            )
            all_scores[seed] = final_scores(history)

        csv_file.close()
        history_file.close()
        print(f"\nper-round per-client eval returns written to {csv_path}")
        print("\n===== final per-client eval return (mean over seeds) =====")
        for client_id in range(cfg.clients_count):
            scores = [all_scores[s][client_id] for s in cfg.seeds if client_id in all_scores[s]]
            mean = statistics.mean(scores)
            std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            print(f"  client{client_id}: {mean:.1f} +/- {std:.1f}")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
