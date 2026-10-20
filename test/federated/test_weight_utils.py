"""Unit tests for tianshou.federated.weight_utils.

These tests run without Ray and validate the weight round-trip, aggregation
math, partial sharing, and the FedProx gradient hook on their own.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy, DQN
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.federated.weight_utils import (
    ClientUpdate,
    FedProxGradientHook,
    extract_fed_state_dict,
    fedavg_aggregate,
    load_fed_state_dict,
)
from tianshou.utils.net.common import Net
from tianshou.utils.space_info import SpaceInfo


def make_cartpole_dqn() -> DQN:
    """Build a small CartPole DQN used by the state-dict tests."""
    env = gym.make("CartPole-v1")
    space_info = SpaceInfo.from_env(env)
    net = Net(
        state_shape=space_info.observation_info.obs_shape,
        action_shape=space_info.action_info.action_shape,
        hidden_sizes=[64, 64],
    )
    policy = DiscreteQLearningPolicy(
        model=net,
        action_space=env.action_space,
        eps_training=0.1,
        eps_inference=0.05,
    )
    env.close()
    return DQN(
        policy=policy,
        optim=AdamOptimizerFactory(lr=1e-3),
        gamma=0.9,
        n_step_return_horizon=3,
        target_update_freq=320,
    )


def test_extract_bypasses_optimizer_state() -> None:
    algo = make_cartpole_dqn()
    payload = extract_fed_state_dict(algo)
    # Algorithm.state_dict would append this key; the payload must not have it.
    assert "_optimizers" not in payload
    assert any(name.startswith("policy.") for name in payload)
    # Payload tensors must be CPU and detached.
    assert all(v.device.type == "cpu" for v in payload.values())


def test_roundtrip_preserves_parameters() -> None:
    algo = make_cartpole_dqn()
    payload = extract_fed_state_dict(algo)
    before = {name: v.detach().clone() for name, v in nn.Module.state_dict(algo).items()}

    # Perturb every parameter, then reload the payload.
    with torch.no_grad():
        for param in algo.parameters():
            param.add_(torch.randn_like(param))
    missing, unexpected = load_fed_state_dict(algo, payload)
    assert missing == [] and unexpected == []

    for name, value in nn.Module.state_dict(algo).items():
        assert torch.allclose(value, before[name]), f"parameter {name} not restored"


def test_partial_aggregation_only_touches_shared_keys() -> None:
    algo = make_cartpole_dqn()
    shared_payload = extract_fed_state_dict(algo, shared_keys=["policy.model"])
    assert all(name.startswith("policy.model.") for name in shared_payload)
    assert shared_payload  # non-empty subset

    # Perturb all parameters and load only the shared subset: shared
    # parameters must return to their original values, non-shared ones must
    # keep the perturbation. (Compare against the payload directly — not
    # against "perturbed - 1.0", whose float32 round-off breaks allclose on
    # near-zero elements.)
    with torch.no_grad():
        for param in algo.parameters():
            param.add_(1.0)
    perturbed = {name: v.detach().clone() for name, v in nn.Module.state_dict(algo).items()}
    load_fed_state_dict(algo, shared_payload)

    for name, value in nn.Module.state_dict(algo).items():
        if name.startswith("policy.model."):
            assert torch.allclose(value, shared_payload[name]), f"{name} not restored"
        else:
            assert torch.allclose(value, perturbed[name]), f"{name} should be untouched"


def test_offpolicy_adapter_excludes_and_hard_syncs_target() -> None:
    from tianshou.federated.adapter import OffPolicyFederatedAdapter

    algo = make_cartpole_dqn()
    adapter = OffPolicyFederatedAdapter(algo)
    payload = adapter.get_shared_state()
    # Target networks must not be shipped; only online weights are shared.
    assert not any("_old." in name for name in payload)
    assert algo.model_old is not None

    # Perturb everything and reload the payload: online weights are restored
    # AND the target network is hard-synced to them (DQN's native full update).
    with torch.no_grad():
        for param in algo.parameters():
            param.add_(1.0)
    adapter.load_shared_state(payload)
    for name, value in nn.Module.state_dict(algo).items():
        if "_old." in name:
            continue  # checked below via the parameter objects
        assert torch.allclose(value, payload[name]), f"{name} not restored"
    for tgt_param, src_param in zip(
        algo.model_old.parameters(), algo.policy.model.parameters(), strict=True
    ):
        assert torch.allclose(tgt_param.data, src_param.data), "target not hard-synced"


def test_fedavg_aggregation_is_step_weighted() -> None:
    a = {"w": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0])}
    b = {"w": torch.tensor([5.0, 0.0]), "b": torch.tensor([1.0])}
    updates = [
        ClientUpdate(client_id=0, weights=a, num_steps=300),
        ClientUpdate(client_id=1, weights=b, num_steps=100),
    ]
    agg = fedavg_aggregate(updates)
    assert torch.allclose(agg["w"], torch.tensor([2.0, 1.5]))  # 0.75a + 0.25b
    assert torch.allclose(agg["b"], torch.tensor([2.5]))

    # Equal steps -> plain mean.
    updates[0].num_steps = updates[1].num_steps
    agg = fedavg_aggregate(updates)
    assert torch.allclose(agg["w"], torch.tensor([3.0, 1.0]))


def test_fedprox_hook_adds_proximal_gradient() -> None:
    module = nn.Linear(2, 1, bias=False)
    mu = 0.1
    hook = FedProxGradientHook(module, mu=mu)

    # Snapshot the global reference, then move the parameters away from it.
    hook.attach_and_snapshot()
    refs = {name: p.detach().clone() for name, p in module.named_parameters()}
    with torch.no_grad():
        for param in module.parameters():
            param.add_(0.3)

    module.weight.grad = None
    loss = module.weight.sum()
    loss.backward()

    for name, param in module.named_parameters():
        expected = torch.ones_like(param) + mu * (param.detach() - refs[name])
        assert param.grad is not None
        assert torch.allclose(param.grad, expected), f"wrong proximal gradient for {name}"

    # Removing the hook restores plain gradients.
    hook.remove()
    module.weight.grad = None
    module.weight.sum().backward()
    assert torch.allclose(module.weight.grad, torch.ones_like(module.weight))


def test_fedprox_hook_respects_shared_prefixes() -> None:
    class TwoParams(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Parameter(torch.tensor([1.0]))
            self.private = nn.Parameter(torch.tensor([2.0]))

    module = TwoParams()
    hook = FedProxGradientHook(module, mu=0.5, shared_prefixes=["shared"])
    hook.attach_and_snapshot()
    with torch.no_grad():
        module.shared.add_(1.0)
        module.private.add_(1.0)
    (module.shared.sum() + module.private.sum()).backward()

    assert torch.allclose(module.shared.grad, torch.tensor([1.5]))  # 1 + 0.5 * 1
    assert torch.allclose(module.private.grad, torch.tensor([1.0]))  # not regularized
    hook.remove()


@pytest.mark.parametrize("mu", [0.0, 0.1, 1.0])
def test_fedprox_hook_attach_remove_roundtrip(mu: float) -> None:
    module = nn.Linear(3, 1)
    hook = FedProxGradientHook(module, mu=mu)
    for _ in range(2):
        hook.attach_and_snapshot()
        (module.weight.sum()).backward()
        hook.remove()
    assert module.weight.grad is not None
    assert np.isfinite(module.weight.grad.numpy()).all()
