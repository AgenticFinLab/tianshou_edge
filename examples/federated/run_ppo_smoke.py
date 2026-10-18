"""Single-machine federated PPO smoke test on CartPole with heterogeneous gravity.

Validates the on-policy federation path end to end: each client collects
fresh trajectories, runs local PPO epochs, uploads shareable weights, and the
coordinator aggregates them with FedAvg.

Run from the repo root (requires the ``ray`` package)::

    python examples/federated/run_ppo_smoke.py --rounds 20 --clients 3
"""
from __future__ import annotations

import argparse

import gymnasium as gym
import torch
from gymnasium import Env
from torch.distributions import Categorical, Distribution

from tianshou.algorithm.modelfree.ppo import PPO
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.federated.adapter import OnPolicyFederatedAdapter
from tianshou.federated.client_actor import remote_client_actor
from tianshou.federated.coordinator import FederatedCoordinator
from tianshou.federated.weight_utils import LocalBudget
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import DiscreteActor, DiscreteCritic
from tianshou.utils.space_info import SpaceInfo


def make_cartpole_env_fn(gravity: float):
    """Create a factory for a CartPole env with modified gravity."""

    def _make() -> Env:
        env = gym.make("CartPole-v1")
        env.gravity = gravity  # client-specific dynamics
        return env

    return _make


def build_ppo_algorithm(lr: float = 1e-3) -> PPO:
    """Build a small discrete PPO for CartPole (CPU); the actor moves it to its device."""
    env = gym.make("CartPole-v1")
    space_info = SpaceInfo.from_env(env)
    obs_shape = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape

    def dist(logits: torch.Tensor) -> Distribution:
        return Categorical(logits=logits)

    actor = DiscreteActor(
        preprocess_net=Net(state_shape=obs_shape, hidden_sizes=[64, 64]),
        action_shape=action_shape,
        softmax_output=False,
    )
    critic = DiscreteCritic(
        preprocess_net=Net(state_shape=obs_shape, hidden_sizes=[64, 64])
    )
    policy = ProbabilisticActorPolicy(
        actor=actor,
        dist_fn=dist,
        action_scaling=False,
        deterministic_eval=True,
        action_space=env.action_space,
    )
    env.close()
    return PPO(
        policy=policy,
        critic=critic,
        optim=AdamOptimizerFactory(lr=lr),
        gamma=0.99,
        gae_lambda=0.95,
        max_batchsize=256,
        eps_clip=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--envs-per-client", type=int, default=4)
    parser.add_argument("--collect", type=int, default=1000,
                        help="fresh env steps collected per client per round")
    parser.add_argument("--epochs", type=int, default=4,
                        help="local PPO epochs over the round's data")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--buffer", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-cpus", type=float, default=1)
    parser.add_argument("--gravity-spread", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import ray  # lazy import so --help works without ray

    ray.init(ignore_reinit_error=True)
    try:
        gravities = [
            9.8 + (i - (args.clients - 1) / 2) * args.gravity_spread
            for i in range(args.clients)
        ]
        print(f"clients: {args.clients}, gravity per client: "
              f"{[round(g, 2) for g in gravities]}")

        actor_cls = remote_client_actor(num_cpus=args.num_cpus)
        clients = [
            actor_cls.remote(
                client_id=i,
                env_fn=make_cartpole_env_fn(gravity),
                algo_builder=lambda: build_ppo_algorithm(lr=args.lr),
                adapter_factory=OnPolicyFederatedAdapter,
                num_train_envs=args.envs_per_client,
                buffer_size=args.buffer,
                device=args.device,
                seed=args.seed,
            )
            for i, gravity in enumerate(gravities)
        ]

        coordinator = FederatedCoordinator(clients)
        history = coordinator.run(
            num_rounds=args.rounds,
            budget=LocalBudget(
                num_env_steps_per_update=args.collect,
                num_updates=args.epochs,
                batch_size=args.batch,
            ),
            eval_every=args.eval_every,
            eval_n_episode=args.eval_episodes,
            seed=args.seed,
        )
        final_mean = (
            list(history[-1]["eval_returns"].values()) if history and history[-1].get("eval_returns") else None
        )
        print(f"smoke test done: {args.rounds} rounds, "
              f"final per-client eval: {final_mean}")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
