"""Configuration schema for the unified federated experiment entry.

A single YAML (or JSON) file drives a complete federated run for DQN, SAC,
or PPO: algorithm, environment, per-client heterogeneity, aggregation, local
training budget, seeds, evaluation frequency, and Ray resources. The entry
script ``examples/federated/run_federated.py`` consumes this schema.

Example (see ``examples/federated/config_examples/``)::

    algorithm: dqn
    env: cartpole
    seeds: [42]
    clients_count: 3
    heterogeneity:
      gravity: [9.3, 9.8, 10.3]
    aggregation: fedavg
    budget:
      rounds: 30
      collect_per_update: 100
      updates_per_round: 10
      batch_size: 64
    evaluation:
      every_rounds: 2
      episodes: 10
    output:
      csv_path: results/dqn_run.csv
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import yaml


@dataclass
class ClientHeterogeneity:
    """Per-client environment parameter values.

    A scalar is broadcast to every client; a list must have one entry per
    client (``clients_count``).
    """

    gravity: list[float] | float = 9.8
    mass: list[float] | float = 1.0
    length: list[float] | float = 1.0
    max_torque: list[float] | float | None = None


@dataclass
class AlgorithmConfig:
    """Network and algorithm hyperparameters; unused fields are ignored per
    algorithm."""

    hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])
    lr: float = 1e-3
    gamma: float = 0.99
    # DQN
    target_update_freq: int = 320
    eps_training: float = 0.1
    n_step: int = 3
    # SAC
    tau: float = 0.005
    alpha: float = 0.2
    # PPO
    eps_clip: float = 0.2
    gae_lambda: float = 0.95
    max_batchsize: int = 256
    vf_coef: float = 0.5
    ent_coef: float = 0.01


@dataclass
class BudgetConfig:
    """Local training budget per round.

    Off-policy: ``collect_per_update`` env steps before each of
    ``updates_per_round`` gradient steps (buffer persists across rounds).
    On-policy: ``collect_per_update`` fresh env steps per round followed by
    ``updates_per_round`` epochs over that round's data (buffer is cleared).
    The parallel env count multiplies the collected steps.
    """

    rounds: int = 20
    collect_per_update: int = 100
    updates_per_round: int = 10
    batch_size: int = 64


@dataclass
class EvaluationConfig:
    every_rounds: int = 2
    episodes: int = 10
    pre_post_eval: bool = False


@dataclass
class RayConfig:
    num_cpus: float = 1
    num_gpus: float = 0


@dataclass
class OutputConfig:
    csv_path: str = "results/federated_run.csv"
    checkpoint_dir: str | None = None


@dataclass
class FederatedConfig:
    """Root schema of a federated experiment."""

    algorithm: str = ""
    env: str = ""
    seeds: list[int] = field(default_factory=lambda: [42])
    device: str = "cpu"
    clients_count: int = 3
    envs_per_client: int = 4
    buffer_size: int = 20000
    warmup_steps: int = 0
    aggregation: str = "fedavg"
    agg_scope: str = "all"
    fedprox_mu: float = 0.0
    heterogeneity: ClientHeterogeneity = field(default_factory=ClientHeterogeneity)
    algorithm_config: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    ray: RayConfig = field(default_factory=RayConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


_VALID = {
    "algorithm": ("dqn", "sac", "ppo"),
    "env": ("cartpole", "pendulum", "halfcheetah", "hopper", "metaworld_reach", "metaworld_push"),
    "aggregation": ("fedavg", "local_only"),
    "agg_scope": ("all", "actor_only"),
}


def load_config(path: str) -> FederatedConfig:
    """Load and validate a federated experiment config file (YAML or JSON).

    :param path: path to the config file; the format is chosen by extension
        (``.yaml``/``.yml`` or ``.json``).
    :raises ValueError: on unknown fields, invalid enum values, or per-client
        heterogeneity lists whose length does not match ``clients_count``.
    """
    with open(path) as f:
        if path.endswith((".yaml", ".yml")):
            raw = yaml.safe_load(f)
        elif path.endswith(".json"):
            raw = json.load(f)
        else:
            raise ValueError(f"unsupported config format for {path!r}; use .yaml or .json")
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    cfg = FederatedConfig(**_pop_known(raw, FederatedConfig, "root"))
    cfg.heterogeneity = ClientHeterogeneity(**raw.get("heterogeneity") or {})
    cfg.algorithm_config = AlgorithmConfig(**raw.get("algorithm_config") or {})
    cfg.budget = BudgetConfig(**raw.get("budget") or {})
    cfg.evaluation = EvaluationConfig(**raw.get("evaluation") or {})
    cfg.ray = RayConfig(**raw.get("ray") or {})
    cfg.output = OutputConfig(**raw.get("output") or {})

    for field_name, allowed in _VALID.items():
        value = getattr(cfg, field_name)
        if value not in allowed:
            raise ValueError(
                f"{field_name} must be one of {allowed}, got {value!r}"
            )
    if cfg.clients_count <= 0 or cfg.aggregation not in _VALID["aggregation"]:
        raise ValueError(f"clients_count must be positive, got {cfg.clients_count}")
    for attr in ("gravity", "mass", "length", "max_torque"):
        value = getattr(cfg.heterogeneity, attr)
        if isinstance(value, list) and len(value) != cfg.clients_count:
            raise ValueError(
                f"heterogeneity.{attr} has {len(value)} entries but "
                f"clients_count is {cfg.clients_count}"
            )
    for seed in cfg.seeds:
        if not isinstance(seed, int):
            raise ValueError(f"seeds must be integers, got {seed!r}")
    return cfg


def _pop_known(raw: dict, dataclass_type: type, where: str) -> dict:
    known = {f.name for f in dataclass_type.__dataclass_fields__.values()}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown fields in {where}: {sorted(unknown)}")
    return raw


def per_client_values(value: list[float] | float, count: int) -> list[float]:
    """Expand a scalar to ``count`` copies, or return the list unchanged."""
    if isinstance(value, list):
        return value
    return [value] * count
