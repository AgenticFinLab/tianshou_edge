"""Shared SAC construction utilities for the federated examples.

Config follows tianshou's official SAC benchmark example
(``examples/mujoco/mujoco_sac.py``): hidden [256, 256], actor/critic lr 1e-3,
gamma 0.99, tau 0.005, fixed alpha 0.2, batch 256, UTD 1, 10k random warm-up.

The twin critics are constructed as two separately initialized instances.
Passing ``critic2=None`` would deepcopy the first critic, keep both networks
identical forever (their targets are the same min), and degenerate the
clipped double-Q mechanism.
"""
from __future__ import annotations

import gymnasium as gym
from gymnasium import Env

from tianshou.algorithm.modelfree.sac import SAC, SACPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ContinuousActorProbabilistic, ContinuousCritic
from tianshou.utils.space_info import SpaceInfo

__all__ = ["make_pendulum_env_fn", "build_pendulum_sac"]


def make_pendulum_env_fn(gravity: float):
    """Create a factory for a Pendulum env with modified gravity (m = l = 1)."""

    def _make() -> Env:
        env = gym.make("Pendulum-v1")
        env.unwrapped.g = gravity
        return env

    return _make


def build_pendulum_sac(
    lr: float = 1e-3,
    hidden_sizes: list[int] | None = None,
    tau: float = 0.005,
    alpha: float = 0.2,
    gravity: float = 10.0,
) -> SAC:
    """Build a Pendulum SAC algorithm and return it (CPU); caller moves it to a device.

    :param gravity: only affects the reference env used to read the action
        space; the dynamics come from the environments created via
        :func:`make_pendulum_env_fn`.
    """
    if hidden_sizes is None:
        hidden_sizes = [256, 256]
    env = gym.make("Pendulum-v1")
    env.unwrapped.g = gravity
    space_info = SpaceInfo.from_env(env)
    obs_shape = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape

    def make_critic() -> ContinuousCritic:
        return ContinuousCritic(
            preprocess_net=Net(
                state_shape=obs_shape,
                action_shape=action_shape,
                hidden_sizes=hidden_sizes,
                concat=True,
            ),
            hidden_sizes=hidden_sizes,
        )

    actor = ContinuousActorProbabilistic(
        preprocess_net=Net(state_shape=obs_shape, hidden_sizes=hidden_sizes),
        action_shape=action_shape,
        hidden_sizes=hidden_sizes,
    )
    critic1 = make_critic()
    critic2 = make_critic()  # independent initialization, no twin degeneracy
    policy = SACPolicy(
        actor=actor,
        action_space=env.action_space,
        action_scaling=True,
        deterministic_eval=True,
    )
    env.close()
    return SAC(
        policy=policy,
        policy_optim=AdamOptimizerFactory(lr=lr),
        critic=critic1,
        critic_optim=AdamOptimizerFactory(lr=lr),
        critic2=critic2,
        critic2_optim=AdamOptimizerFactory(lr=lr),
        tau=tau,
        gamma=0.99,
        alpha=alpha,
    )
