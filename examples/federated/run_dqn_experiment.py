"""DQN federated comparison: Local-only vs FedAvg vs FedProx on heterogeneous CartPole.

The purpose of this experiment is infrastructure validation, not algorithm
research: it proves that federated training actually learns (greedy
evaluation returns rising from ~10 towards 200+, the CartPole random-policy
score being ~10) and that aggregation does not silently break training.

Protocol (identical for all methods and seeds):

* 3 clients, each with its own CartPole environment (gravity 9.8 +/- spread),
  same training budget per round: 100 collected env steps then 10 gradient
  steps, same replay buffer size, same seed set.
* Evaluation every ``--eval-every`` rounds: each client runs
  ``--eval-episodes`` complete greedy episodes (eps_inference = 0) on its OWN
  environment; per-client returns are recorded per round. Training-collector
  returns are biased towards short episodes and are NOT used as the learning
  signal.
* Final statistics are computed per seed as the mean of per-client means over
  the last 10 evaluation rounds, then aggregated across seeds as mean +/- std
  (with seeds as the statistical unit, not evaluation points).
* Local-only runs the same driver-driven round structure with identity
  aggregation (no communication).

Run from the repo root::

    python examples/federated/run_dqn_experiment.py --rounds 100 \
        --seeds 42 43 44 --out /tmp/dqn_fed_experiment.csv

Requires the ``ray`` package and the ``frl_cuda`` conda env (see repo notes).
"""
from __future__ import annotations

import argparse
import csv
import statistics

import gymnasium as gym
from gymnasium import Env

from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy, DQN
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.federated.adapter import OffPolicyFederatedAdapter
from tianshou.federated.client_actor import remote_client_actor
from tianshou.federated.coordinator import FederatedCoordinator
from tianshou.federated.weight_utils import LocalBudget
from tianshou.utils.net.common import Net
from tianshou.utils.space_info import SpaceInfo


def make_cartpole_env_fn(gravity: float):
    """Create a factory for a CartPole env with modified gravity."""

    def _make() -> Env:
        env = gym.make("CartPole-v1")
        env.gravity = gravity  # client-specific dynamics
        return env

    return _make


def build_dqn_algorithm(lr: float = 1e-3) -> DQN:
    """Build a small DQN for CartPole (CPU); the actor moves it to its device.

    eps_training=0.1 keeps exploration alive during local training; the
    evaluation path runs outside the training-step context, where the policy
    uses eps_inference=0.0 (fully greedy).
    """
    env = gym.make("CartPole-v1")
    space_info = SpaceInfo.from_env(env)
    state_shape = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape
    net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[64, 64])
    policy = DiscreteQLearningPolicy(
        model=net,
        action_space=env.action_space,
        eps_training=0.1,
        eps_inference=0.0,
    )
    env.close()
    return DQN(
        policy=policy,
        optim=AdamOptimizerFactory(lr=lr),
        gamma=0.9,
        n_step_return_horizon=3,
        target_update_freq=320,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--envs-per-client", type=int, default=4)
    parser.add_argument("--collect", type=int, default=100,
                        help="env steps collected per client per local update")
    parser.add_argument("--updates", type=int, default=10,
                        help="local gradient steps per client per round")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--buffer", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mu", type=float, default=0.05,
                        help="FedProx proximal term strength (FedProx method only)")
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="complete greedy episodes per client per evaluation")
    parser.add_argument("--methods", nargs="+", default=["local_only", "fedavg", "fedprox"],
                        choices=["local_only", "fedavg", "fedprox"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="independent experiment seeds")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-cpus", type=float, default=1,
                        help="CPUs requested per client actor")
    parser.add_argument("--gravity-spread", type=float, default=0.5,
                        help="gravity offset between consecutive clients")
    parser.add_argument("--out", type=str, default=None,
                        help="CSV file for per-round per-client eval returns")
    return parser.parse_args()


def build_clients(args: argparse.Namespace, mu: float, seed: int):
    """Create the client actors; identical for all methods (fair comparison)."""
    gravities = [
        9.8 + (i - (args.clients - 1) / 2) * args.gravity_spread
        for i in range(args.clients)
    ]
    actor_cls = remote_client_actor(num_cpus=args.num_cpus)
    return [
        actor_cls.remote(
            client_id=i,
            env_fn=make_cartpole_env_fn(gravity),
            algo_builder=lambda: build_dqn_algorithm(lr=args.lr),
            adapter_factory=OffPolicyFederatedAdapter,
            num_train_envs=args.envs_per_client,
            buffer_size=args.buffer,
            device=args.device,
            fedprox_mu=mu,
            seed=seed,
        )
        for i, gravity in enumerate(gravities)
    ]


def eval_rows(method: str, seed: int, history: list[dict]) -> list[tuple]:
    """Flatten a run's history into (method, seed, round, client_id, return) rows."""
    rows: list[tuple] = []
    for summary in history:
        for client_id, mean_return in summary.get("eval_returns", {}).items():
            rows.append((method, seed, summary["round"], client_id, mean_return))
    return rows


def final_score(history: list[dict], last_n: int = 10) -> float:
    """Final score of one run.

    Per client: mean over the last ``last_n`` evaluation rounds (each round =
    ``eval_episodes`` greedy episodes). Then averaged over clients. Returns
    NaN if no evaluation happened.
    """
    per_client: dict[int, list[float]] = {}
    for summary in history:
        for client_id, mean_return in summary.get("eval_returns", {}).items():
            per_client.setdefault(client_id, []).append(mean_return)
    if not per_client or not any(per_client.values()):
        return float("nan")
    client_means = [statistics.mean(returns[-last_n:]) for returns in per_client.values()]
    return float(statistics.mean(client_means))


def run_method(args: argparse.Namespace, method: str, seed: int) -> list[dict]:
    aggregation = "fedavg" if method in ("fedavg", "fedprox") else "local_only"
    mu = args.mu if method == "fedprox" else 0.0
    clients = build_clients(args, mu=mu, seed=seed)
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
        verbose=True,
    )


def main() -> None:
    args = parse_args()
    import ray  # lazy import so --help works without ray

    ray.init(ignore_reinit_error=True)
    try:
        print(
            f"clients={args.clients} rounds={args.rounds} budget={args.collect} steps/"
            f"{args.updates} updates per round, eval {args.eval_episodes} episodes every "
            f"{args.eval_every} rounds, seeds={args.seeds}"
        )
        all_rows: list[tuple] = []
        per_seed_scores: dict[tuple[str, int], float] = {}
        for seed in args.seeds:
            for method in args.methods:
                print(f"\n===== seed {seed} | method: {method} =====")
                history = run_method(args, method, seed)
                score = final_score(history)
                per_seed_scores[(method, seed)] = score
                all_rows.extend(eval_rows(method, seed, history))

        print("\n===== comparison =====")
        print("final score = per-client mean over last 10 eval rounds, pooled over clients")
        print(f"{'method':11s} {'per-seed scores':>30s}  {'mean ± std':>12s}")
        method_scores: dict[str, list[float]] = {}
        for method in args.methods:
            scores = [per_seed_scores[(method, seed)] for seed in args.seeds]
            method_scores[method] = scores
            scores_str = ", ".join(f"{s:.1f}" for s in scores)
            mean, std = statistics.mean(scores), statistics.stdev(scores) if len(scores) > 1 else 0.0
            print(f"{method:11s} {scores_str:>30s}  {mean:8.1f} ± {std:5.1f}")
        if "fedavg" in method_scores and "local_only" in method_scores:
            deltas = [
                fed - local
                for fed, local in zip(method_scores["fedavg"], method_scores["local_only"], strict=True)
            ]
            print(f"delta FedAvg-LocalOnly: mean={statistics.mean(deltas):+.1f}")

        if args.out:
            with open(args.out, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["method", "seed", "round", "client_id", "eval_return"])
                writer.writerows(all_rows)
            print(f"\nper-round per-client eval returns written to {args.out}")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
