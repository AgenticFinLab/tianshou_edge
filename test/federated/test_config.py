"""Unit tests for tianshou.federated.config."""
from __future__ import annotations

import pytest

from tianshou.federated.config import FederatedConfig, load_config, per_client_values


def _write(path, text: str) -> None:
    path.write_text(text)


def test_load_valid_yaml(tmp_path) -> None:
    p = tmp_path / "cfg.yaml"
    _write(p, "algorithm: dqn\nenv: cartpole\nseeds: [42, 43]\nclients_count: 2\n"
              "heterogeneity:\n  gravity: [9.3, 9.8]\n")
    cfg = load_config(str(p))
    assert cfg.algorithm == "dqn"
    assert cfg.seeds == [42, 43]
    assert cfg.budget.rounds == 20  # defaults apply


def test_load_json(tmp_path) -> None:
    p = tmp_path / "cfg.json"
    _write(p, '{"algorithm": "sac", "env": "pendulum", "seeds": [42]}')
    cfg = load_config(str(p))
    assert cfg.algorithm == "sac"
    assert cfg.heterogeneity.gravity == 9.8  # scalar default


def test_unknown_field_rejected(tmp_path) -> None:
    p = tmp_path / "cfg.yaml"
    _write(p, "algorithm: dqn\nenv: cartpole\nbogus_field: 1\n")
    with pytest.raises(ValueError, match="unknown fields"):
        load_config(str(p))


def test_invalid_enum_rejected(tmp_path) -> None:
    p = tmp_path / "cfg.yaml"
    _write(p, "algorithm: rainbow\nenv: cartpole\n")
    with pytest.raises(ValueError, match="algorithm must be one of"):
        load_config(str(p))


def test_heterogeneity_length_mismatch_rejected(tmp_path) -> None:
    p = tmp_path / "cfg.yaml"
    _write(p, "algorithm: dqn\nenv: cartpole\nclients_count: 4\n"
              "heterogeneity:\n  gravity: [9.3, 9.8]\n")
    with pytest.raises(ValueError, match="heterogeneity.gravity"):
        load_config(str(p))


def test_unsupported_format_rejected(tmp_path) -> None:
    p = tmp_path / "cfg.toml"
    _write(p, "algorithm = 'dqn'\n")
    with pytest.raises(ValueError, match="unsupported config format"):
        load_config(str(p))


def test_per_client_values_expansion() -> None:
    assert per_client_values(9.8, 3) == [9.8, 9.8, 9.8]
    assert per_client_values([1.0, 2.0, 3.0], 3) == [1.0, 2.0, 3.0]


def test_default_config_is_empty_but_loadable() -> None:
    cfg = FederatedConfig()
    assert cfg.algorithm == ""
    assert cfg.seeds == [42]
