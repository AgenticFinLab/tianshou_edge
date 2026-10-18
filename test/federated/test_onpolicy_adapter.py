"""Unit tests for tianshou.federated.adapter.OnPolicyFederatedAdapter."""
from __future__ import annotations

import gymnasium as gym
import torch
from torch import nn
from torch.distributions import Categorical, Distribution

from tianshou.algorithm.modelfree.ppo import PPO
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.federated.adapter import OnPolicyFederatedAdapter
from tianshou.federated.weight_utils import LocalBudget, load_fed_state_dict
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import DiscreteActor, DiscreteCritic
from tianshou.utils.space_info import SpaceInfo


def make_cartpole_ppo() -> tuple[PPO, DummyVectorEnv, Collector, VectorReplayBuffer]:
    """Build a small discrete PPO on CartPole plus a working collector."""
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
        action_space=env.action_space,
    )
    env.close()
    algo = PPO(
        policy=policy,
        critic=critic,
        optim=AdamOptimizerFactory(lr=1e-3),
        gamma=0.99,
        gae_lambda=0.95,
        max_batchsize=256,
        eps_clip=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
    )
    envs = DummyVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(4)])
    buffer = VectorReplayBuffer(20000, buffer_num=4)
    collector = Collector(algo, envs, buffer, exploration_noise=False)
    collector.reset()
    return algo, envs, collector, buffer


def test_payload_scopes_and_ret_rms_exclusion() -> None:
    algo, _, _, _ = make_cartpole_ppo()

    adapter_all = OnPolicyFederatedAdapter(algo, agg_scope="all")
    payload_all = adapter_all.get_shared_state()
    assert "_optimizers" not in payload_all
    assert any(name.startswith("policy.actor.") for name in payload_all)
    assert any(name.startswith("critic.") for name in payload_all)
    # Reward-scale-dependent running statistics never enter the payload.
    assert not any(name.startswith("ret_rms.") for name in payload_all)

    adapter_actor = OnPolicyFederatedAdapter(algo, agg_scope="actor_only")
    payload_actor = adapter_actor.get_shared_state()
    assert payload_actor and all(
        name.startswith(OnPolicyFederatedAdapter.ACTOR_PREFIX) for name in payload_actor
    )


def test_actor_only_loading_leaves_critic_and_ret_rms_untouched() -> None:
    algo, _, _, _ = make_cartpole_ppo()
    adapter = OnPolicyFederatedAdapter(algo, agg_scope="actor_only")
    payload = adapter.get_shared_state()

    with torch.no_grad():
        for param in algo.parameters():
            param.add_(1.0)
    perturbed = {name: v.detach().clone() for name, v in nn.Module.state_dict(algo).items()}
    load_fed_state_dict(algo, payload)

    for name, value in nn.Module.state_dict(algo).items():
        if name.startswith("policy.actor."):
            assert torch.allclose(value, payload[name]), f"{name} not restored"
        else:
            assert torch.allclose(value, perturbed[name]), f"{name} should be untouched"


def test_local_train_round_lifecycle() -> None:
    algo, envs, collector, buffer = make_cartpole_ppo()
    adapter = OnPolicyFederatedAdapter(algo, agg_scope="all")
    budget = LocalBudget(num_env_steps_per_update=400, num_updates=4, batch_size=64)

    stats = adapter.local_train(collector, buffer, budget)

    assert stats.num_steps > 0
    # On-policy data is consumed within the round: the buffer must be empty
    # afterwards so the next round starts from fresh trajectories.
    assert len(buffer) == 0
    # A second round must run cleanly (regression guard for the lifecycle).
    stats2 = adapter.local_train(collector, buffer, budget)
    assert stats2.num_steps > 0
    assert len(buffer) == 0


def test_prox_prefixes_restricted_to_actor() -> None:
    algo, _, _, _ = make_cartpole_ppo()
    for agg_scope in ("all", "actor_only"):
        adapter = OnPolicyFederatedAdapter(algo, agg_scope=agg_scope)
        assert adapter.prox_prefixes == ["policy.actor."]
