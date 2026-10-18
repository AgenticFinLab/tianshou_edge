"""Single-machine federated DQN smoke test on CartPole with heterogeneous dynamics.

Each client runs its own CartPole environment with a different gravity
constant, trains a DQN locally, and ships weights to a coordinator actor which
aggregates them with FedAvg every round. This validates the full FL closed
loop (extract -> broadcast -> local train -> aggregate) before anything
algorithmic is built on top of it.

Run from the repo root (requires the `ray` package)::

    python examples/federated/run_dqn_smoke.py --rounds 50 --clients 3

Optional FedProx proximal term: ``--mu 0.1``.
"""
from __future__ import annotations

import argparse

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
    """Build a small DQN for CartPole (CPU); the actor moves it to its device."""
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
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--envs-per-client", type=int, default=4)
    parser.add_argument("--collect", type=int, default=100,
                        help="env steps collected per client per local update")
    parser.add_argument("--updates", type=int, default=10,
                        help="local gradient steps per client per round")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--buffer", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mu", type=float, default=0.0,
                        help="FedProx proximal term strength; 0 disables it")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-cpus", type=float, default=1,
                        help="CPUs requested per client actor")
    parser.add_argument("--gravity-spread", type=float, default=0.5,
                        help="gravity offset between consecutive clients")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import ray  # lazy import so --help works without ray

    ray.init(ignore_reinit_error=True)
    try:
        # Heterogeneous dynamics: clients sit around the default gravity 9.8.
        gravities = [
            9.8 + (i - (args.clients - 1) / 2) * args.gravity_spread
            for i in range(args.clients)
        ]
        print(f"clients: {args.clients}, gravity per client: "
              f"{[round(g, 2) for g in gravities]}, fedprox mu: {args.mu}")

        actor_cls = remote_client_actor(num_cpus=args.num_cpus)
        clients = [
            actor_cls.remote(
                client_id=i,
                env_fn=make_cartpole_env_fn(gravity),
                algo_builder=lambda: build_dqn_algorithm(lr=args.lr),
                adapter_factory=OffPolicyFederatedAdapter,
                num_train_envs=args.envs_per_client,
                buffer_size=args.buffer,
                device=args.device,
                fedprox_mu=args.mu,
                seed=42,
            )
            for i, gravity in enumerate(gravities)
        ]

        coordinator = FederatedCoordinator(clients)
        history = coordinator.run(
            num_rounds=args.rounds,
            budget=LocalBudget(
                num_env_steps_per_update=args.collect,
                num_updates=args.updates,
                batch_size=args.batch,
            ),
        )
        final_mean = history[-1]["mean_return"] if history else None
        print(f"smoke test done: {args.rounds} rounds, "
              f"final mean_return={final_mean}")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
