"""Unit tests for tianshou.federated.persistence and full-state restore."""
from __future__ import annotations

import pickle

import numpy as np
import torch

from tianshou.federated.persistence import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip(tmp_path) -> None:
    global_weights = {"w": torch.tensor([1.0, 2.0])}
    client_states = [{"client_id": 0, "weights": {"w": torch.tensor([3.0])}}]
    save_checkpoint(tmp_path / "ckpt", {"next_round": 6}, global_weights, client_states)

    manifest, loaded_weights, loaded_states = load_checkpoint(tmp_path / "ckpt")
    assert manifest == {"next_round": 6}
    assert torch.allclose(loaded_weights["w"], global_weights["w"])
    assert loaded_states == client_states


def test_checkpoint_overwrites_previous_round(tmp_path) -> None:
    save_checkpoint(tmp_path / "ckpt", {"next_round": 2}, {"w": torch.tensor([1.0])}, [])
    save_checkpoint(tmp_path / "ckpt", {"next_round": 3}, {"w": torch.tensor([2.0])}, [])
    manifest, loaded_weights, _ = load_checkpoint(tmp_path / "ckpt")
    assert manifest["next_round"] == 3
    assert torch.allclose(loaded_weights["w"], torch.tensor([2.0]))


def test_missing_checkpoint_raises(tmp_path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope")


def test_algorithm_state_pickle_preserves_optimizer_state() -> None:
    """A full algorithm state dict (weights + _optimizers) survives pickling,
    which is what the client actor ships for full-state checkpointing."""
    from test.federated.test_weight_utils import make_cartpole_dqn

    algo = make_cartpole_dqn()
    # Advance the optimizer so its state is non-trivial.
    from tianshou.data import Collector, VectorReplayBuffer
    from tianshou.env import DummyVectorEnv
    from tianshou.utils.torch_utils import policy_within_training_step, torch_train_mode

    buffer = VectorReplayBuffer(10000, buffer_num=4)
    collector = Collector(algo, DummyVectorEnv(
        [lambda: __import__("gymnasium").make("CartPole-v1") for _ in range(4)]),
        buffer)
    collector.reset()
    collector.collect(n_step=200, random=True)
    with policy_within_training_step(algo.policy), torch_train_mode(algo):
        algo.update(sample_size=64, buffer=buffer)

    state = algo.state_dict()
    opt_state_before = state["_optimizers"][0]["state"]
    assert opt_state_before, "optimizer state should be non-empty after an update"

    restored = pickle.loads(pickle.dumps(state))

    def assert_state_equal(a, b, where: str = "") -> None:
        if isinstance(a, torch.Tensor):
            assert isinstance(b, torch.Tensor), f"{where}: tensor lost"
            assert torch.allclose(a, b), f"{where}: tensor value changed"
        elif isinstance(a, dict):
            assert a.keys() == b.keys(), f"{where}: dict keys differ"
            for k in a:
                assert_state_equal(a[k], b[k], f"{where}.{k}")
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b), f"{where}: sequence length differs"
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                assert_state_equal(x, y, f"{where}[{i}]")
        else:
            assert a == b, f"{where}: scalar changed"

    assert_state_equal(restored, state, "algorithm_state")
