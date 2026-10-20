"""Checkpoint persistence for federated runs.

A checkpoint directory holds:

* ``manifest.json`` — run metadata (algorithm/env echo, seeds, the round to
  resume from),
* ``global_weights.pkl`` — the aggregated shared weights,
* ``client_states.pkl`` — one full client state per client (weights,
  optimizer states, replay buffer, random streams) as produced by
  ``TianshouFedClientActor.get_full_state``.

Resuming supports two modes:

* ``"full"`` — restores weights, optimizers, buffers, and RNG; an exact
  continuation of the clients.
* ``"model_only"`` — restores the aggregated weights only; clients restart
  with fresh buffers/optimizers/RNG and continue training from the model.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

__all__ = ["save_checkpoint", "load_checkpoint"]


def save_checkpoint(
    checkpoint_dir: str | Path,
    manifest: dict,
    global_weights: dict,
    client_states: list[dict],
) -> None:
    """Write a checkpoint directory.

    :param manifest: JSON-serializable metadata (must include the round to
        resume from, e.g. ``{"next_round": 6}``).
    :param global_weights: aggregated shared weights (CPU tensors).
    :param client_states: full state of every client, in client-id order.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with open(checkpoint_dir / "global_weights.pkl", "wb") as f:
        pickle.dump(global_weights, f)
    with open(checkpoint_dir / "client_states.pkl", "wb") as f:
        pickle.dump(client_states, f)


def load_checkpoint(
    checkpoint_dir: str | Path,
) -> tuple[dict, dict, list[dict]]:
    """Load a checkpoint written by :func:`save_checkpoint`.

    :return: ``(manifest, global_weights, client_states)``.
    :raises FileNotFoundError: if the directory or a required file is missing.
    """
    checkpoint_dir = Path(checkpoint_dir)
    manifest = json.loads((checkpoint_dir / "manifest.json").read_text())
    with open(checkpoint_dir / "global_weights.pkl", "rb") as f:
        global_weights = pickle.load(f)
    with open(checkpoint_dir / "client_states.pkl", "rb") as f:
        client_states = pickle.load(f)
    return manifest, global_weights, client_states
