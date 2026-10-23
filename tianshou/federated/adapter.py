"""Per-algorithm federated adapters.

Different algorithms organize their module state differently (DQN has online
plus target Q nets, SAC has an actor plus critics, PPO has a policy plus a
value net). The adapter is the single place that declares, per algorithm:

* which parameters participate in federation (``shared_keys``) — everything by
  default; restricting to e.g. ``["policy.actor"]`` enables partial
  aggregation / personalization under environment heterogeneity,
* how target networks are treated — the off-policy adapter excludes them from
  the payload and re-syncs them locally after aggregation (hard copy for DQN,
  which is its native update; soft-update algorithms need a dedicated
  adapter),
* how a local training round is executed (off-policy vs on-policy loop).

The server/driver only ever sees normalized payloads produced by
:func:`~tianshou.federated.weight_utils.extract_fed_state_dict` and never
touches algorithm internals directly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch

from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.data import ReplayBuffer
from tianshou.data.collector import Collector
from tianshou.federated.weight_utils import (
    LocalBudget,
    LocalTrainStats,
    extract_fed_state_dict,
    load_fed_state_dict,
)
from tianshou.utils.torch_utils import policy_within_training_step, torch_train_mode

__all__ = [
    "FederatedAlgorithmAdapter",
    "OffPolicyFederatedAdapter",
    "SACFederatedAdapter",
    "OnPolicyFederatedAdapter",
]


class FederatedAlgorithmAdapter(ABC):
    """Declares how one algorithm participates in a federated round."""

    def __init__(self, algorithm: Algorithm) -> None:
        self.algorithm = algorithm

    @property
    def shared_keys(self) -> list[str] | None:
        """Parameter-name prefixes that participate in federation.

        ``None`` means all parameters are shared.
        """
        return None

    @property
    def exclude_target_net(self) -> bool:
        """Whether target-network parameters are excluded from the payload.

        If ``True``, the client is responsible for re-syncing its target
        networks locally after aggregation.
        """
        return False

    @property
    def exclude_prefixes(self) -> list[str] | None:
        """Parameter-name prefixes that never enter the shared payload.

        Reserved for state that must stay client-local even in full
        aggregation, e.g. SAC's entropy temperature (``alpha.*``), whose scale
        is reward-scale dependent.
        """
        return None

    @property
    def prox_prefixes(self) -> list[str] | None:
        """Parameter-name prefixes the FedProx proximal term is applied to.

        Defaults to ``shared_keys`` (regularize everything that is shared);
        algorithms with value networks (SAC) restrict it to the policy to
        avoid pulling the critic against its non-stationary TD targets.
        """
        return self.shared_keys

    def get_shared_state(self) -> dict[str, torch.Tensor]:
        """Return the shareable weights of the wrapped algorithm (CPU tensors)."""
        return extract_fed_state_dict(
            self.algorithm,
            shared_keys=self.shared_keys,
            exclude_target_net=self.exclude_target_net,
            exclude_prefixes=self.exclude_prefixes,
        )

    def load_shared_state(self, state: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
        """Load aggregated weights into the wrapped algorithm.

        :return: ``(missing_keys, unexpected_keys)`` for key-mismatch
            diagnosis; loading is tolerant of missing keys so that partial
            aggregation works.
        """
        return load_fed_state_dict(self.algorithm, state)

    @abstractmethod
    def local_train(
        self,
        collector: Collector,
        buffer: ReplayBuffer,
        budget: LocalBudget,
    ) -> LocalTrainStats:
        """Run one round of local training against the wrapped algorithm.

        Called after :meth:`load_shared_state`. Must respect tianshou 2.0.1's
        training-step context (see :func:`policy_within_training_step`).
        """


class OffPolicyFederatedAdapter(FederatedAlgorithmAdapter):
    """Local training loop for off-policy algorithms (DQN family, SAC, TD3...).

    The replay buffer persists across rounds, so experiences accumulated by
    the client never leave it — only weights are exchanged.

    Target networks are excluded from the shared payload (halving its size);
    after the aggregated online weights are loaded, targets are re-synced
    locally according to ``target_sync_mode``:

    * ``"hard"`` — exact copy of the online weights into the targets. This
      matches DQN's native full target update (``LaggedNetworkFullUpdate``
      mixin) and is the standard FedDQN rule; it only resets the local
      target-update counter.
    * ``"keep_local"`` — leave the client's targets untouched (they keep
      following the loaded weights via the algorithm's own periodic updates).

    For soft-update algorithms (SAC/TD3), use a dedicated adapter with Polyak
    alignment instead — a hard reset at every round would disrupt the
    Bellman-backup stability there.
    """

    def __init__(
        self,
        algorithm: Algorithm,
        target_sync_mode: str = "hard",
        exclude_target_net: bool = True,
        exclude_prefixes: list[str] | None = None,
    ) -> None:
        super().__init__(algorithm)
        if target_sync_mode not in ("hard", "keep_local"):
            raise ValueError(
                f"unknown target_sync_mode {target_sync_mode!r}, expected 'hard' or 'keep_local'"
            )
        self.target_sync_mode = target_sync_mode
        self._exclude_target_net = exclude_target_net
        self._exclude_prefixes = exclude_prefixes

    @property
    def exclude_target_net(self) -> bool:
        return self._exclude_target_net

    @property
    def exclude_prefixes(self) -> list[str] | None:
        return self._exclude_prefixes

    def load_shared_state(self, state: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
        missing, unexpected = super().load_shared_state(state)
        if self.target_sync_mode == "hard" and hasattr(self.algorithm, "_lagged_networks"):
            # LaggedNetworkCollection.full_parameter_update copies every online
            # parameter into its lagged (target) counterpart — the same
            # mechanism DQN itself uses for its periodic target updates.
            self.algorithm._lagged_networks.full_parameter_update()
        return missing, unexpected

    def local_train(
        self,
        collector: Collector,
        buffer: ReplayBuffer,
        budget: LocalBudget,
    ) -> LocalTrainStats:
        total_steps = 0
        episode_returns: list[float] = []
        # Both collection and updates must happen inside a training step so
        # that the policy knows it may explore (eps-greedy etc.).
        with policy_within_training_step(self.algorithm.policy):
            for _ in range(budget.num_updates):
                collect_stats = collector.collect(n_step=budget.num_env_steps_per_update)
                total_steps += collect_stats.n_collected_steps
                if collect_stats.n_collected_episodes > 0:
                    episode_returns.extend(collect_stats.returns.tolist())
                # Collector restores the module mode after collection; make
                # sure the update runs in train mode (matters for e.g. BatchNorm).
                with torch_train_mode(self.algorithm):
                    self.algorithm.update(sample_size=budget.batch_size, buffer=buffer)
        mean_return = float(np.mean(episode_returns)) if episode_returns else None
        return LocalTrainStats(num_steps=total_steps, mean_return=mean_return)


class SACFederatedAdapter(OffPolicyFederatedAdapter):
    """Federated adapter for SAC (continuous control).

    Module map in tianshou 2.0.1 (see ``tianshou/algorithm/modelfree/sac.py``):

    * ``policy.actor.*`` — the stochastic policy; shared in every scope,
    * ``critic.*`` / ``critic2.*`` — twin Q-functions; shared only in
      ``agg_scope="all"``,
    * ``critic_old.*`` / ``critic2_old.*`` — lagged target critics; in
      ``agg_scope="all"`` they are included in the FedAvg payload, which keeps
      the aggregation consistent with Polyak linearity. They are **never**
      hard-reset after loading (SAC's soft update with tau=0.005 is what keeps
      the Bellman backup stable; ``target_sync_mode`` is therefore forced to
      ``"keep_local"``),
    * ``alpha.*`` — the entropy temperature; always client-local because its
      scale depends on the local reward magnitude, which differs across
      heterogeneous clients.

    ``agg_scope="actor_only"`` shares only the actor and leaves critics,
    target critics, and alpha private. Under environment heterogeneity the
    true value functions differ across clients, so averaging critics can pull
    them into conflict; whether actor-only sharing is actually better is an
    experimental question, not an assumption — it is provided as the
    research-motivated comparison to the full-sharing baseline.
    """

    ACTOR_PREFIX = "policy.actor."

    def __init__(self, algorithm: Algorithm, agg_scope: str = "all") -> None:
        if agg_scope not in ("all", "actor_only", "critic_only"):
            raise ValueError(
                f"unknown agg_scope {agg_scope!r}, expected 'all', 'actor_only', "
                f"or 'critic_only'"
            )
        super().__init__(
            algorithm,
            target_sync_mode="keep_local",
            exclude_target_net=False,
            exclude_prefixes=["alpha."],
        )
        self.agg_scope = agg_scope

    @property
    def shared_keys(self) -> list[str] | None:
        if self.agg_scope == "actor_only":
            return [self.ACTOR_PREFIX]
        if self.agg_scope == "critic_only":
            # Share the online critics only; target critics and the actor
            # stay client-local (targets re-track via the local Polyak
            # update as training proceeds).
            return ["critic.", "critic2."]
        return None

    @property
    def prox_prefixes(self) -> list[str]:
        # Pulling the critics towards the previous round's global weights
        # fights their non-stationary TD targets (which must grow with the
        # return scale); the proximal term applies to the actor only.
        return [self.ACTOR_PREFIX]


class OnPolicyFederatedAdapter(FederatedAlgorithmAdapter):
    """Federated adapter for on-policy actor-critic algorithms (PPO, A2C).

    On-policy rounds follow the collect -> multi-epoch update -> clear
    lifecycle: each round collects fresh trajectories, performs
    ``budget.num_updates`` epochs over them (``algorithm.update(repeat=...)``),
    then resets the buffer. This differs from the off-policy loop on purpose —
    the adapter does not flatten the two regimes into one interface.

    Module map in tianshou 2.0.1 (PPO extends A2C):

    * ``policy.actor.*`` — the stochastic policy; shared in every scope,
    * ``critic.*`` — the value network; shared only in ``agg_scope="all"``,
    * ``ret_rms.*`` — running statistics of observed returns; reward-scale
      dependent, so always client-local (like SAC's alpha).

    PPO has no target networks, so no target-sync logic applies.
    """

    ACTOR_PREFIX = "policy.actor."

    def __init__(self, algorithm: Algorithm, agg_scope: str = "all") -> None:
        if agg_scope not in ("all", "actor_only"):
            raise ValueError(
                f"unknown agg_scope {agg_scope!r}, expected 'all' or 'actor_only'"
            )
        super().__init__(algorithm)
        self.agg_scope = agg_scope

    @property
    def shared_keys(self) -> list[str] | None:
        if self.agg_scope == "actor_only":
            return [self.ACTOR_PREFIX]
        return None

    @property
    def exclude_prefixes(self) -> list[str]:
        return ["ret_rms."]

    @property
    def prox_prefixes(self) -> list[str]:
        return [self.ACTOR_PREFIX]

    def local_train(
        self,
        collector: Collector,
        buffer: ReplayBuffer,
        budget: LocalBudget,
    ) -> LocalTrainStats:
        """Run one on-policy round: collect fresh data, update for
        ``budget.num_updates`` epochs, then clear the buffer.

        ``budget.num_env_steps_per_update`` is the number of fresh environment
        steps collected per round (the parallel env count multiplies it).
        """
        with policy_within_training_step(self.algorithm.policy):
            collect_stats = collector.collect(n_step=budget.num_env_steps_per_update)
            with torch_train_mode(self.algorithm):
                self.algorithm.update(
                    buffer=buffer,
                    batch_size=budget.batch_size,
                    repeat=budget.num_updates,
                )
        # On-policy data is consumed within the round; the buffer is cleared
        # (statistics kept) so the next round starts from fresh trajectories.
        buffer.reset(keep_statistics=True)
        returns = collect_stats.returns.tolist()
        mean_return = float(np.mean(returns)) if returns else None
        return LocalTrainStats(
            num_steps=collect_stats.n_collected_steps,
            mean_return=mean_return,
        )
