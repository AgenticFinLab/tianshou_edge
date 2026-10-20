"""Unit tests for tianshou.federated.adapter.SACFederatedAdapter."""
from __future__ import annotations

import gymnasium as gym
import torch
from torch import nn

from tianshou.algorithm.modelfree.sac import AutoAlpha, SAC, SACPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.federated.adapter import SACFederatedAdapter
from tianshou.federated.weight_utils import load_fed_state_dict
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ContinuousActorProbabilistic, ContinuousCritic
from tianshou.utils.space_info import SpaceInfo


def make_pendulum_sac(alpha: float | AutoAlpha = 0.2) -> SAC:
    """Build a small SAC for Pendulum-v1 (CPU)."""
    env = gym.make("Pendulum-v1")
    space_info = SpaceInfo.from_env(env)
    obs_shape = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape
    actor = ContinuousActorProbabilistic(
        preprocess_net=Net(state_shape=obs_shape, hidden_sizes=[64, 64]),
        action_shape=action_shape,
        hidden_sizes=[64, 64],
    )
    critic = ContinuousCritic(
        preprocess_net=Net(
            state_shape=obs_shape,
            action_shape=action_shape,
            hidden_sizes=[64, 64],
            concat=True,
        ),
        hidden_sizes=[64, 64],
    )
    policy = SACPolicy(actor=actor, action_space=env.action_space, deterministic_eval=True)
    env.close()
    return SAC(
        policy=policy,
        policy_optim=AdamOptimizerFactory(lr=3e-4),
        critic=critic,
        critic_optim=AdamOptimizerFactory(lr=3e-4),
        alpha=alpha,
    )


def test_agg_scope_all_includes_critics_and_targets() -> None:
    algo = make_pendulum_sac()
    adapter = SACFederatedAdapter(algo, agg_scope="all")
    payload = adapter.get_shared_state()

    assert "_optimizers" not in payload
    assert any(name.startswith("policy.actor.") for name in payload)
    assert any(name.startswith("critic.") for name in payload)
    assert any(name.startswith("critic2.") for name in payload)
    # Target critics are part of the FedAvg payload so that the aggregation
    # stays consistent with Polyak linearity (no hard reset anywhere).
    assert any(name.startswith("critic_old.") for name in payload)
    assert any(name.startswith("critic2_old.") for name in payload)


def test_agg_scope_actor_only_shares_only_actor() -> None:
    algo = make_pendulum_sac()
    adapter = SACFederatedAdapter(algo, agg_scope="actor_only")
    payload = adapter.get_shared_state()

    assert payload and all(
        name.startswith(SACFederatedAdapter.ACTOR_PREFIX) for name in payload
    )

    # Perturb everything and load the actor-only payload: the actor is
    # restored, critics and targets keep the perturbation.
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


def test_alpha_never_enters_payload() -> None:
    # AutoAlpha is a registered nn.Module, so its parameter shows up in
    # nn.Module.state_dict — the adapter must exclude it in every scope.
    target_entropy = -1.0  # -dim(action) for Pendulum
    algo = make_pendulum_sac(alpha=AutoAlpha(target_entropy, 0.0, AdamOptimizerFactory(lr=3e-4)))
    raw_keys = list(nn.Module.state_dict(algo))
    assert any(name.startswith("alpha.") for name in raw_keys), "test setup: alpha in state dict"

    for agg_scope in ("all", "actor_only"):
        adapter = SACFederatedAdapter(algo, agg_scope=agg_scope)
        payload = adapter.get_shared_state()
        assert not any(name.startswith("alpha.") for name in payload), agg_scope


def test_no_hard_target_sync_on_load() -> None:
    algo = make_pendulum_sac()
    adapter = SACFederatedAdapter(algo, agg_scope="all")
    assert adapter.target_sync_mode == "keep_local"

    # Make the target critics diverge from the online critics, then snapshot.
    with torch.no_grad():
        for name, param in algo.named_parameters():
            if "_old." in name:
                param.add_(0.5)
    payload = adapter.get_shared_state()

    # Perturb the online critics only and reload: targets must return to their
    # snapshot values (NOT be hard-copied from the online critics).
    with torch.no_grad():
        for name, param in algo.named_parameters():
            if name.startswith("critic.") or name.startswith("critic2."):
                param.add_(1.0)
    adapter.load_shared_state(payload)

    for name, value in nn.Module.state_dict(algo).items():
        if "_old." in name:
            assert torch.allclose(value, payload[name]), f"{name} hard-reset by the load!"
        assert torch.allclose(value, payload[name]), f"{name} not restored"


def test_prox_prefixes_restricted_to_actor() -> None:
    algo = make_pendulum_sac()
    for agg_scope in ("all", "actor_only"):
        adapter = SACFederatedAdapter(algo, agg_scope=agg_scope)
        assert adapter.prox_prefixes == ["policy.actor."]
