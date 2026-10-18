"""Ray actor hosting a single federated client.

Each client owns its own vectorized environment(s), replay buffer, algorithm
instance, and optimizer state. Raw transitions never leave the client — only
shareable weights (see :mod:`tianshou.federated.weight_utils`) are exchanged
with the coordinator.

The class is intentionally defined **without** a ``@ray.remote`` decorator so
that per-client resources can be chosen at construction time via
:func:`remote_client_actor` (e.g. ``num_cpus=1, num_gpus=0.25``).
"""
from __future__ import annotations

import pickle
import random
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from gymnasium import Env

from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.data import VectorReplayBuffer
from tianshou.data.collector import Collector
from tianshou.env.venvs import DummyVectorEnv
from tianshou.federated.adapter import FederatedAlgorithmAdapter
from tianshou.federated.weight_utils import ClientUpdate, FedProxGradientHook, LocalBudget

__all__ = ["TianshouFedClientActor", "remote_client_actor"]

EnvFn = Callable[[], Env]
AlgoBuilder = Callable[[], Algorithm]
AdapterFactory = Callable[[Algorithm], FederatedAlgorithmAdapter]


def remote_client_actor(
    num_cpus: float = 1,
    num_gpus: float = 0,
) -> Any:
    """Return a Ray actor class wrapping :class:`TianshouFedClientActor`.

    Ray is imported lazily so that this module (and the rest of
    ``tianshou.federated``) stays importable without Ray installed.
    """
    import ray  # lazy import, see RayVectorEnv in tianshou.env for the precedent

    return ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)(TianshouFedClientActor)


class TianshouFedClientActor:
    """One federated client: local env + algorithm + buffer + collector."""

    def __init__(
        self,
        *,
        client_id: int,
        env_fn: EnvFn,
        algo_builder: AlgoBuilder,
        adapter_factory: AdapterFactory,
        num_train_envs: int = 4,
        num_test_envs: int = 5,
        buffer_size: int = 10000,
        warmup_steps: int = 0,
        device: str | torch.device | None = None,
        fedprox_mu: float = 0.0,
        seed: int = 0,
    ) -> None:
        """
        :param client_id: unique identifier of this client.
        :param env_fn: factory creating one local environment instance; the
            clients can differ here to model environment heterogeneity (e.g.
            different dynamics).
        :param algo_builder: factory creating the client's algorithm instance
            (CPU). The actor moves it to ``device`` afterwards.
        :param adapter_factory: builds the :class:`FederatedAlgorithmAdapter`
            that defines sharing and the local training loop.
        :param num_train_envs: number of parallel training envs of this client
            (a :class:`~tianshou.env.DummyVectorEnv` is created).
        :param num_test_envs: number of parallel evaluation envs used by
            :meth:`evaluate`.
        :param buffer_size: capacity of the client's persistent replay buffer.
        :param warmup_steps: environment steps collected with uniformly random
            actions before the first round (buffer warm-up, matching the
            single-process baseline protocol).
        :param device: device to run this client's algorithm on; defaults to
            CUDA when available.
        :param fedprox_mu: strength of the FedProx proximal term; ``<= 0``
            disables it.
        :param seed: base random seed; the client id is added to it so that
            clients do not share random streams.
        """
        self.client_id = client_id
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # Give each client an independent random stream.
        seed = seed + client_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.train_envs = DummyVectorEnv([env_fn for _ in range(num_train_envs)])
        self.algorithm = algo_builder().to(self.device)
        self.adapter = adapter_factory(self.algorithm)
        self.buffer = VectorReplayBuffer(buffer_size, buffer_num=num_train_envs)
        self.collector = Collector(
            self.algorithm,
            self.train_envs,
            self.buffer,
            exploration_noise=True,
        )
        # Evaluation collector: complete episodes, no exploration. Because
        # evaluation happens outside policy_within_training_step, the policy
        # acts with its inference epsilon (0.0 in the examples) instead of the
        # training epsilon.
        self.test_envs = DummyVectorEnv([env_fn for _ in range(num_test_envs)])
        self.test_collector = Collector(self.algorithm, self.test_envs, exploration_noise=True)
        # The first collect() requires a reset, otherwise the collector has no
        # cached initial observation yet (tianshou/data/collector.py:831).
        self.collector.reset()
        self.test_collector.reset()
        if warmup_steps > 0:
            # Random-action warm-up (no policy involved), identical to the
            # single-process baseline's collect(n_step=..., random=True).
            self.collector.collect(n_step=warmup_steps, random=True)
        self.prox_hook: FedProxGradientHook | None = None
        if fedprox_mu > 0:
            self.prox_hook = FedProxGradientHook(
                self.algorithm,
                mu=fedprox_mu,
                # The adapter declares which parameters the proximal term may
                # pull on (e.g. SAC restricts it to the actor).
                shared_prefixes=self.adapter.prox_prefixes,
            )

    def get_client_id(self) -> int:
        """Return this client's identifier."""
        return self.client_id

    def get_initial_weights(self) -> dict[str, torch.Tensor]:
        """Return this client's shareable weights before any training."""
        return self.adapter.get_shared_state()

    def evaluate(self, n_episode: int = 5) -> dict[str, object]:
        """Evaluate the current policy on complete local episodes.

        Called outside the training-step context, so the policy explores with
        its inference epsilon only. Training statistics (``res.returns`` of
        the training collector) would bias towards short, completed episodes,
        which is why evaluation has its own collector.

        :param n_episode: number of complete episodes to run.
        :return: ``{"mean_return": float, "returns": list[float]}``.
        """
        stats = self.test_collector.collect(n_episode=n_episode)
        returns = stats.returns.tolist()
        return {
            "mean_return": float(np.mean(returns)) if returns else float("nan"),
            "returns": returns,
        }

    def sync_weights(self, global_weights: dict[str, torch.Tensor]) -> None:
        """Load aggregated weights into the local algorithm (no training)."""
        self.adapter.load_shared_state(global_weights)

    def local_train(self, budget: LocalBudget) -> ClientUpdate:
        """Train locally for one round and report; weights must be synced first.

        Split from :meth:`sync_weights` so the coordinator can evaluate the
        client right after the sync (pre/post-aggregation diagnostics).
        """
        if self.prox_hook is not None:
            # Snapshot the current (just-synced) weights as the FedProx reference.
            self.prox_hook.attach_and_snapshot()
        try:
            stats = self.adapter.local_train(self.collector, self.buffer, budget)
        finally:
            if self.prox_hook is not None:
                self.prox_hook.remove()
        return ClientUpdate(
            client_id=self.client_id,
            weights=self.adapter.get_shared_state(),
            num_steps=stats.num_steps,
            mean_return=stats.mean_return,
        )

    def local_train_round(
        self,
        global_weights: dict[str, torch.Tensor],
        budget: LocalBudget,
    ) -> ClientUpdate:
        """Run one federated round: sync global weights, train locally, report.

        Convenience wrapper around :meth:`sync_weights` + :meth:`local_train`.

        :param global_weights: aggregated weights broadcast by the server.
        :param budget: how much local work to perform this round.
        :return: the client's updated shareable weights and round statistics.
        """
        self.sync_weights(global_weights)
        return self.local_train(budget)

    def get_full_state(self) -> dict:
        """Return this client's full training state for checkpointing.

        Includes shareable weights, optimizer states (momentum), the replay
        buffer, and the random streams. The returned dict is picklable.
        """
        return {
            "client_id": self.client_id,
            "weights": self.adapter.get_shared_state(),
            # nn.Module.state_dict bypasses the "_optimizers" key, so the
            # full Algorithm.state_dict is used here — it additionally
            # carries the optimizer states.
            "algorithm_state": self.algorithm.state_dict(),
            "buffer": pickle.dumps(self.buffer),
            "rng": {
                "random": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
        }

    def load_full_state(self, state: dict, mode: str = "full") -> None:
        """Restore a checkpointed state produced by :meth:`get_full_state`.

        :param state: the pickled state dict.
        :param mode: ``"full"`` restores weights, optimizer states, replay
            buffer, and random streams (exact continuation of the client);
            ``"model_only"`` restores the shareable weights only (fresh
            buffer/optimizers/RNG — the training continues from the model).
        """
        if mode not in ("full", "model_only"):
            raise ValueError(f"unknown mode {mode!r}, expected 'full' or 'model_only'")
        self.adapter.load_shared_state(state["weights"])
        if mode == "full":
            self.algorithm.load_state_dict(state["algorithm_state"])
            new_buffer = pickle.loads(state["buffer"])
            self.buffer = new_buffer
            self.collector.buffer = new_buffer
            random.setstate(state["rng"]["random"])
            np.random.set_state(state["rng"]["numpy"])
            torch.set_rng_state(state["rng"]["torch"])
