"""Utilities for extracting, aggregating, and loading federated model weights.

All functions operate on plain ``dict[str, torch.Tensor]`` payloads that can be
shipped through Ray's object store. Payloads are always kept on CPU so that
clients with heterogeneous devices can consume them.

Design notes for tianshou 2.0.1 (see ``tianshou/algorithm/algorithm_base.py``):

* ``Algorithm.state_dict()`` appends an ``"_optimizers"`` key, and
  ``Algorithm.load_state_dict()`` pops it **without a default**, so a pure
  FedAvg payload would raise ``KeyError``. We therefore bypass the overridden
  methods by calling ``torch.nn.Module.state_dict`` / ``load_state_dict``
  directly, which keeps optimizer state (momentum, etc.) local to each client.
* Target networks (e.g. DQN's ``model_old``) are registered submodules and
  appear in the state dict. By default they are included in the payload and
  aggregated like any other parameter; aggregating them is consistent because
  soft (Polyak) updates are linear in the parameters. Hard-resetting target
  networks after aggregation would disrupt the update schedule, so we never do
  that here. Alternatively, pass ``exclude_target_net=True`` and re-sync
  targets locally after loading.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

__all__ = [
    "LocalBudget",
    "LocalTrainStats",
    "ClientUpdate",
    "extract_fed_state_dict",
    "load_fed_state_dict",
    "fedavg_aggregate",
    "FedProxGradientHook",
]


@dataclass
class LocalBudget:
    """Work a client performs in one federated round.

    :param num_env_steps_per_update: environment steps collected before each
        local update (the number of parallel envs multiplies this).
    :param num_updates: number of local update iterations per round. For
        off-policy algorithms this is the number of gradient steps; for
        on-policy algorithms it denotes local epochs over the collected data.
    :param batch_size: transitions sampled per gradient step (off-policy only).
    """

    num_env_steps_per_update: int
    num_updates: int
    batch_size: int = 64


@dataclass
class LocalTrainStats:
    """Statistics gathered while a client trains locally for one round.

    :param num_steps: total environment steps collected during the round.
    :param mean_return: mean episodic return over episodes finished during the
        round, or ``None`` if no episode completed.
    """

    num_steps: int
    mean_return: float | None = None


@dataclass
class ClientUpdate:
    """Result of one client's local training round.

    :param client_id: identifier of the producing client.
    :param weights: shared model weights after local training (CPU tensors).
    :param num_steps: total environment steps collected during the round;
        used as the aggregation weight.
    :param mean_return: mean episodic return over episodes finished during the
        round, or ``None`` if no episode completed.
    """

    client_id: int
    weights: dict[str, torch.Tensor]
    num_steps: int
    mean_return: float | None = None


def extract_fed_state_dict(
    algo: nn.Module,
    shared_keys: list[str] | None = None,
    exclude_target_net: bool = False,
    exclude_prefixes: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Extract the shareable model weights of a tianshou ``Algorithm``.

    Bypasses ``Algorithm.state_dict`` on purpose (see module docstring), so the
    payload never contains the ``"_optimizers"`` key.

    :param algo: the algorithm instance.
    :param shared_keys: if given, only parameters whose names start with one of
        these prefixes are extracted. This enables partial aggregation /
        personalization (e.g. ``["policy.actor"]`` to share only the actor).
    :param exclude_target_net: if True, parameters of target networks (keys
        containing ``"_old."`` or ``"target"``) are excluded. The caller is
        then responsible for re-syncing targets locally after aggregation.
    :param exclude_prefixes: parameters whose names start with one of these
        prefixes are never extracted, even if they match ``shared_keys``.
        Useful for state that must stay client-local (e.g. SAC's entropy
        temperature ``alpha.*``).
    :return: a CPU-only mapping of parameter names to detached tensors.
    """
    # nn.Module.state_dict called unbound on the instance runs the base
    # implementation, skipping Algorithm.state_dict's "_optimizers" extension.
    raw_sd = nn.Module.state_dict(algo)
    fed_sd: dict[str, torch.Tensor] = {}
    for name, value in raw_sd.items():
        if exclude_target_net and ("_old." in name or "target" in name):
            continue
        if exclude_prefixes is not None and any(
            name.startswith(p) for p in exclude_prefixes
        ):
            continue
        if shared_keys is not None and not any(name.startswith(p) for p in shared_keys):
            continue
        fed_sd[name] = value.detach().cpu().clone()
    return fed_sd


def load_fed_state_dict(
    algo: nn.Module,
    fed_sd: dict[str, torch.Tensor],
) -> tuple[list[str], list[str]]:
    """Load a shared-weight payload into an algorithm.

    Uses ``nn.Module.load_state_dict`` (not ``Algorithm.load_state_dict``) to
    avoid the mandatory ``"_optimizers"`` key and to keep local optimizer
    state untouched. Tensor values are copied into the existing parameters, so
    the algorithm's current device is respected.

    :param algo: the algorithm instance.
    :param fed_sd: payload produced by :func:`extract_fed_state_dict` or by
        :func:`fedavg_aggregate`. May be a subset of the algorithm's state
        (partial aggregation).
    :return: ``(missing_keys, unexpected_keys)`` reported by the underlying
        ``load_state_dict`` call; useful for diagnosing key mismatches.
    """
    # strict=False tolerates partial aggregation (shared_keys subset);
    # the returned lists make silent key mismatches visible to the caller.
    return nn.Module.load_state_dict(algo, fed_sd, strict=False)


def fedavg_aggregate(updates: list[ClientUpdate]) -> dict[str, torch.Tensor]:
    """Aggregate client updates with FedAvg, weighted by collected env steps.

    :param updates: per-client round results. All payloads must share the same
        keys.
    :return: aggregated weights on CPU with the dtype of the first client's
        tensors.
    """
    if not updates:
        raise ValueError("fedavg_aggregate requires at least one client update")
    total_steps = sum(u.num_steps for u in updates)
    if total_steps <= 0:
        raise ValueError("total collected steps must be positive")
    first_weights = updates[0].weights
    aggregated: dict[str, torch.Tensor] = {}
    for name in first_weights:
        ref = first_weights[name]
        acc = torch.zeros_like(ref, dtype=torch.float32)
        for update in updates:
            acc.add_(update.weights[name].float(), alpha=update.num_steps / total_steps)
        aggregated[name] = acc.to(ref.dtype)
    return aggregated


class FedProxGradientHook:
    """Zero-intrusion FedProx regularizer for tianshou algorithms.

    FedProx adds a proximal term mu/2 * ||theta - theta_global||^2 to the local
    training objective, which amounts to adding ``mu * (theta - theta_global)``
    to each parameter gradient. PyTorch's ``register_hook`` on parameters fires
    after ``backward()`` and before ``optimizer.step()``, so the regularizer
    can be injected without touching tianshou's ``Algorithm.Optimizer.step``
    (which fuses zero_grad + backward + step) or any other source file.

    Usage per round::

        hook.attach_and_snapshot()   # after loading the global weights
        ... local training ...
        hook.remove()

    :param module: the algorithm (or any nn.Module) to regularize.
    :param mu: proximal term strength. ``mu <= 0`` disables the hook.
    :param shared_prefixes: if given, only parameters whose names start with
        one of these prefixes are regularized (should match ``shared_keys``).
    """

    def __init__(
        self,
        module: nn.Module,
        mu: float = 0.0,
        shared_prefixes: list[str] | None = None,
    ) -> None:
        self.module = module
        self.mu = mu
        self.shared_prefixes = shared_prefixes
        self._global_refs: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def attach_and_snapshot(self) -> None:
        """Snapshot the current parameters as theta_global and arm the hooks."""
        self.remove()
        if self.mu <= 0:
            return
        for name, param in self.module.named_parameters():
            if not param.requires_grad or "_old." in name:
                continue
            if self.shared_prefixes and not any(
                name.startswith(prefix) for prefix in self.shared_prefixes
            ):
                continue
            self._global_refs[name] = param.detach().clone()
            # Bind `param` and `ref` via default arguments: a closure over the
            # loop variables would late-bind and all hooks would read the last
            # parameter of the iteration.
            self._handles.append(
                param.register_hook(
                    self._make_hook(param, self._global_refs[name], self.mu)
                )
            )

    def remove(self) -> None:
        """Remove all registered hooks and the snapshot."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._global_refs.clear()

    @staticmethod
    def _make_hook(
        param: torch.Tensor,
        global_ref: torch.Tensor,
        mu: float,
    ) -> "callable[[torch.Tensor], torch.Tensor]":
        def _hook(grad: torch.Tensor) -> torch.Tensor:
            return grad.add(param.detach() - global_ref, alpha=mu)

        return _hook
