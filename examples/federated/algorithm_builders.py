"""Shared environment and algorithm builders for the federated examples.

Single source of truth for the DQN (CartPole), SAC (Pendulum), and PPO
(CartPole) constructions used by the unified config-driven entry
(``run_federated.py``) and the standalone smoke scripts.
"""
from __future__ import annotations

import gymnasium as gym
import torch
from gymnasium import Env
from torch.distributions import Categorical, Distribution

from examples.federated.sac_builder import build_pendulum_sac, make_pendulum_env_fn
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy, DQN
from tianshou.algorithm.modelfree.ppo import PPO
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import DiscreteActor, DiscreteCritic
from tianshou.utils.space_info import SpaceInfo

__all__ = [
    "make_cartpole_env_fn",
    "make_pendulum_env_fn",
    "build_dqn_cartpole",
    "build_sac_pendulum",
    "build_ppo_cartpole",
    "get_env_fn",
    "get_algo_builder",
]


def make_cartpole_env_fn(gravity: float):
    """Create a factory for a CartPole env with modified gravity."""

    def _make() -> Env:
        env = gym.make("CartPole-v1")
        env.gravity = gravity
        return env

    return _make


def build_dqn_cartpole(
    lr: float = 1e-3,
    hidden_sizes: list[int] | None = None,
    gamma: float = 0.99,
    n_step: int = 3,
    target_update_freq: int = 320,
    eps_training: float = 0.1,
) -> DQN:
    """Build a small DQN for CartPole (CPU); the caller moves it to a device.

    eps_inference is 0.0: evaluation runs outside the training-step context,
    so the policy acts fully greedy there.
    """
    if hidden_sizes is None:
        hidden_sizes = [64, 64]
    env = gym.make("CartPole-v1")
    space_info = SpaceInfo.from_env(env)
    state_shape = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape
    net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=hidden_sizes)
    policy = DiscreteQLearningPolicy(
        model=net,
        action_space=env.action_space,
        eps_training=eps_training,
        eps_inference=0.0,
    )
    env.close()
    return DQN(
        policy=policy,
        optim=AdamOptimizerFactory(lr=lr),
        gamma=gamma,
        n_step_return_horizon=n_step,
        target_update_freq=target_update_freq,
    )


def build_sac_pendulum(
    lr: float = 1e-3,
    hidden_sizes: list[int] | None = None,
    tau: float = 0.005,
    alpha: float = 0.2,
    gravity: float = 10.0,
):
    """Build a Pendulum SAC (delegates to :mod:`examples.federated.sac_builder`)."""
    return build_pendulum_sac(lr=lr, hidden_sizes=hidden_sizes, tau=tau, alpha=alpha,
                              gravity=gravity)


def build_ppo_cartpole(
    lr: float = 1e-3,
    hidden_sizes: list[int] | None = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    max_batchsize: int = 256,
    eps_clip: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> PPO:
    """Build a small discrete PPO for CartPole (CPU)."""
    if hidden_sizes is None:
        hidden_sizes = [64, 64]
    env = gym.make("CartPole-v1")
    space_info = SpaceInfo.from_env(env)
    obs_shape = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape

    def dist(logits: torch.Tensor) -> Distribution:
        return Categorical(logits=logits)

    actor = DiscreteActor(
        preprocess_net=Net(state_shape=obs_shape, hidden_sizes=hidden_sizes),
        action_shape=action_shape,
        softmax_output=False,
    )
    critic = DiscreteCritic(
        preprocess_net=Net(state_shape=obs_shape, hidden_sizes=hidden_sizes)
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
        gamma=gamma,
        gae_lambda=gae_lambda,
        max_batchsize=max_batchsize,
        eps_clip=eps_clip,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
    )


def get_env_fn(env: str, heterogeneity: dict) -> callable:
    """Return an env factory for one client from the config heterogeneity."""
    if env == "cartpole":
        return make_cartpole_env_fn(heterogeneity["gravity"])
    if env == "pendulum":
        fn = make_pendulum_env_fn(heterogeneity["gravity"])
        if heterogeneity.get("mass") != 1.0 or heterogeneity.get("length") != 1.0:

            def _make_with_physics():
                e = fn()
                e.unwrapped.m = heterogeneity["mass"]
                e.unwrapped.l = heterogeneity["length"]
                if heterogeneity.get("max_torque") is not None:
                    e.unwrapped.max_torque = heterogeneity["max_torque"]
                return e

            return _make_with_physics
        return fn
    raise ValueError(f"unknown env {env!r}")


def get_algo_builder(algorithm: str, algo_cfg) -> callable:
    """Return an algorithm builder closure from the config."""
    if algorithm == "dqn":
        return lambda: build_dqn_cartpole(
            lr=algo_cfg.lr,
            hidden_sizes=algo_cfg.hidden_sizes,
            gamma=algo_cfg.gamma,
            n_step=algo_cfg.n_step,
            target_update_freq=algo_cfg.target_update_freq,
            eps_training=algo_cfg.eps_training,
        )
    if algorithm == "sac":
        return lambda: build_sac_pendulum(
            lr=algo_cfg.lr,
            hidden_sizes=algo_cfg.hidden_sizes,
            tau=algo_cfg.tau,
            alpha=algo_cfg.alpha,
        )
    if algorithm == "ppo":
        return lambda: build_ppo_cartpole(
            lr=algo_cfg.lr,
            hidden_sizes=algo_cfg.hidden_sizes,
            gamma=algo_cfg.gamma,
            gae_lambda=algo_cfg.gae_lambda,
            max_batchsize=algo_cfg.max_batchsize,
            eps_clip=algo_cfg.eps_clip,
            vf_coef=algo_cfg.vf_coef,
            ent_coef=algo_cfg.ent_coef,
        )
    raise ValueError(f"unknown algorithm {algorithm!r}")
