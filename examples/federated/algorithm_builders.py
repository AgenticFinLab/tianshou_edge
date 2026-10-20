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
from tianshou.algorithm.modelfree.sac import SAC, SACPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ContinuousActorProbabilistic, ContinuousCritic
from tianshou.utils.net.discrete import DiscreteActor, DiscreteCritic
from tianshou.utils.space_info import SpaceInfo

__all__ = [
    "make_cartpole_env_fn",
    "make_pendulum_env_fn",
    "make_mujoco_env_fn",
    "make_metaworld_env_fn",
    "build_dqn_cartpole",
    "build_sac_pendulum",
    "build_sac_mujoco",
    "build_sac_metaworld",
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


def make_mujoco_env_fn(task: str):
    """Create a factory for a plain MuJoCo env (identical task per client).

    Used by the platform benchmarks: clients run identical tasks, so no
    physics modifications are applied here. Heterogeneous MuJoCo clients
    (mass/friction perturbations) are future work.
    """

    def _make() -> Env:
        return gym.make(task)

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


def make_metaworld_env_fn(task: str):
    """Create a factory for a Meta-World task env (fresh goals per reset)."""

    def _make() -> Env:
        import metaworld

        mt1 = metaworld.MT1(task)
        env = mt1.train_classes[task]()
        # metaworld internal: train envs must be unfrozen to sample fresh
        # goals, and set_task must be called before the first step.
        env._freeze_rand_vec = False
        env.set_task(mt1.train_tasks[0])
        return env

    return _make


def build_sac_metaworld(
    task: str = "reach-v3",
    lr: float = 1e-3,
    hidden_sizes: list[int] | None = None,
    tau: float = 0.005,
    alpha: float = 0.2,
):
    """Build a SAC for a Meta-World task (CPU); the caller moves it to a device."""
    if hidden_sizes is None:
        hidden_sizes = [256, 256]
    env = make_metaworld_env_fn(task)()
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
    policy = SACPolicy(actor=actor, action_space=env.action_space, deterministic_eval=True)
    env.close()
    return SAC(
        policy=policy,
        policy_optim=AdamOptimizerFactory(lr=lr),
        critic=make_critic(),
        critic_optim=AdamOptimizerFactory(lr=lr),
        critic2=make_critic(),
        critic2_optim=AdamOptimizerFactory(lr=lr),
        tau=tau,
        gamma=0.99,
        alpha=alpha,
    )


def build_sac_mujoco(
    task: str = "HalfCheetah-v5",
    lr: float = 1e-3,
    hidden_sizes: list[int] | None = None,
    tau: float = 0.005,
    alpha: float = 0.2,
):
    """Build a SAC for a MuJoCo task (CPU); the caller moves it to a device.

    Follows the tianshou MuJoCo SAC benchmark config (examples/mujoco/
    mujoco_sac.py): independent twin critics, tanh-squashed Gaussian actor.
    """
    if hidden_sizes is None:
        hidden_sizes = [256, 256]
    env = gym.make(task)
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
    policy = SACPolicy(actor=actor, action_space=env.action_space, deterministic_eval=True)
    env.close()
    return SAC(
        policy=policy,
        policy_optim=AdamOptimizerFactory(lr=lr),
        critic=make_critic(),
        critic_optim=AdamOptimizerFactory(lr=lr),
        critic2=make_critic(),
        critic2_optim=AdamOptimizerFactory(lr=lr),
        tau=tau,
        gamma=0.99,
        alpha=alpha,
    )


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


_MUJOCO_TASKS = {"halfcheetah": "HalfCheetah-v5", "hopper": "Hopper-v5"}
_METAWORLD_TASKS = {"metaworld_reach": "reach-v3", "metaworld_push": "push-v3"}


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
    if env in _MUJOCO_TASKS:
        # Benchmark clients run identical tasks; heterogeneity for MuJoCo
        # (mass/friction perturbations) is future work.
        return make_mujoco_env_fn(_MUJOCO_TASKS[env])
    if env in _METAWORLD_TASKS:
        return make_metaworld_env_fn(_METAWORLD_TASKS[env])
    raise ValueError(f"unknown env {env!r}")


def get_algo_builder(algorithm: str, env: str, algo_cfg) -> callable:
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
        if env in _MUJOCO_TASKS:
            task = _MUJOCO_TASKS[env]
            return lambda: build_sac_mujoco(
                task=task,
                lr=algo_cfg.lr,
                hidden_sizes=algo_cfg.hidden_sizes,
                tau=algo_cfg.tau,
                alpha=algo_cfg.alpha,
            )
        if env in _METAWORLD_TASKS:
            task = _METAWORLD_TASKS[env]
            return lambda: build_sac_metaworld(
                task=task,
                lr=algo_cfg.lr,
                hidden_sizes=algo_cfg.hidden_sizes,
                tau=algo_cfg.tau,
                alpha=algo_cfg.alpha,
            )
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
