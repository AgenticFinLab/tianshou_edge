"""Driver-driven federated round orchestration.

The coordinator explicitly triggers every client round and performs the
aggregation; clients never block inside callbacks waiting on the server. This
keeps the control flow at the driver, which makes partial participation and
straggler handling trivial (e.g. swap ``ray.get`` for ``ray.wait`` later).

Two aggregation modes are supported:

* ``"fedavg"`` — clients' shared weights are averaged (weighted by collected
  env steps) and broadcast to everyone each round.
* ``"local_only"`` — each client keeps training on its own weights; the same
  driver-driven loop is used with an identity aggregation. This provides the
  no-federation baseline with an identical training budget.

Requires Ray (imported lazily so the rest of ``tianshou.federated`` stays
importable without it).
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from tianshou.federated.weight_utils import ClientUpdate, LocalBudget, fedavg_aggregate

__all__ = ["FederatedCoordinator"]


class FederatedCoordinator:
    """Runs federated rounds over a set of Ray client actors.

    :param clients: handles to remote :class:`TianshouFedClientActor`
        instances (see :func:`tianshou.federated.client_actor.remote_client_actor`).
    """

    def __init__(self, clients: Sequence[Any]) -> None:
        import ray  # lazy import, see module docstring

        if not clients:
            raise ValueError("FederatedCoordinator requires at least one client")
        self._ray = ray
        self.clients = clients
        self.client_ids = self._ray.get([c.get_client_id.remote() for c in clients])
        # Bootstrap: for FedAvg, the global model starts from an arbitrary
        # client (all clients are built from the same algorithm factory). For
        # local-only training, each client keeps its own weights.
        initial_weights = self._ray.get([c.get_initial_weights.remote() for c in clients])
        self.global_weights: dict[str, Any] = initial_weights[0]
        self.weights_by_client: dict[int, dict[str, Any]] = dict(
            zip(self.client_ids, initial_weights, strict=True)
        )

    def run(
        self,
        num_rounds: int,
        budget: LocalBudget,
        clients_per_round: int | None = None,
        aggregation: str = "fedavg",
        eval_every: int | None = None,
        eval_n_episode: int = 5,
        seed: int = 0,
        verbose: bool = True,
        on_round: Any | None = None,
        eval_pre_post: bool = False,
        post_eval_updates: int = 0,
        checkpoint_dir: str | None = None,
        resume_from: str | None = None,
        resume_mode: str = "full",
    ) -> list[dict[str, Any]]:
        """Run ``num_rounds`` federated rounds.

        :param num_rounds: number of communication rounds.
        :param budget: local work performed by each client per round.
        :param clients_per_round: number of clients sampled per round
            (partial participation); ``None`` uses all clients.
        :param aggregation: ``"fedavg"`` or ``"local_only"`` (see module
            docstring).
        :param eval_every: evaluate every ``eval_every`` rounds on all
            clients; ``None`` disables evaluation.
        :param eval_n_episode: complete episodes per client per evaluation.
        :param seed: seed for the client sampling RNG.
        :param verbose: print a summary line per round.
        :param on_round: optional callback invoked with each round's summary
            right after it is produced; enables streaming results to disk so
            an interrupted run loses at most the current round.
        :param eval_pre_post: if True, evaluate each selected client right
            before the weight sync and right after it (before local
            training), recording ``eval_pre`` / ``eval_post`` per client in
            the summary. Off by default: it doubles the evaluation traffic.
        :param post_eval_updates: with ``eval_pre_post``, also run this many
            gradient updates after the sync and evaluate again, recording
            ``eval_post_k``. This separates the immediate shock (post) from
            the delayed effect that reaches the policy through gradient
            coupling (post_k); the updates count toward the round budget.
        :param checkpoint_dir: if given, a checkpoint is written after every
            round (full client state; see
            :mod:`tianshou.federated.persistence`).
        :param resume_from: checkpoint directory to resume from; rounds are
            counted from the checkpoint's ``next_round``.
        :param resume_mode: ``"full"`` or ``"model_only"`` restore mode (see
            :func:`~tianshou.federated.persistence.load_checkpoint`).
        :return: per-round summaries; evaluation results are stored per client
            under the ``"eval_returns"`` key.
        """
        if aggregation not in ("fedavg", "local_only"):
            raise ValueError(f"unknown aggregation {aggregation!r}")
        from tianshou.federated.persistence import load_checkpoint, save_checkpoint

        start_round = 1
        if resume_from is not None:
            manifest, global_weights, client_states = load_checkpoint(resume_from)
            self.global_weights = global_weights
            sync_futures = [
                client.sync_weights.remote(global_weights) for client in self.clients
            ]
            self._ray.get(sync_futures)
            load_futures = [
                client.load_full_state.remote(state, mode=resume_mode)
                for client, state in zip(self.clients, client_states, strict=True)
            ]
            self._ray.get(load_futures)
            start_round = int(manifest["next_round"])

        rng = random.Random(seed)
        history: list[dict[str, Any]] = []
        for round_idx in range(start_round, num_rounds + 1):
            selected_indices = self._sample_client_indices(clients_per_round, rng)

            if aggregation == "fedavg":
                weights_ref = self._ray.put(self.global_weights)
                per_client_refs = [weights_ref] * len(selected_indices)
            else:
                per_client_refs = [
                    self._ray.put(self.weights_by_client[self.client_ids[i]])
                    for i in selected_indices
                ]

            probe_steps = 0
            if eval_pre_post:
                pre = self._evaluate_selected(selected_indices, eval_n_episode)
                self._ray.get([
                    self.clients[i].sync_weights.remote(ref)
                    for i, ref in zip(selected_indices, per_client_refs, strict=True)
                ])
                post = self._evaluate_selected(selected_indices, eval_n_episode)
                post_k = None
                if 0 < post_eval_updates < budget.num_updates:
                    # Delayed-shock probe: train a fraction of the round
                    # budget, evaluate, then finish the round normally.
                    k_budget = LocalBudget(
                        num_env_steps_per_update=budget.num_env_steps_per_update,
                        num_updates=post_eval_updates,
                        batch_size=budget.batch_size,
                    )
                    k_updates = self._ray.get([
                        self.clients[i].local_train.remote(k_budget) for i in selected_indices
                    ])
                    probe_steps = sum(u.num_steps for u in k_updates)
                    post_k = self._evaluate_selected(selected_indices, eval_n_episode)
                    rest_budget = LocalBudget(
                        num_env_steps_per_update=budget.num_env_steps_per_update,
                        num_updates=budget.num_updates - post_eval_updates,
                        batch_size=budget.batch_size,
                    )
                    updates = self._ray.get([
                        self.clients[i].local_train.remote(rest_budget)
                        for i in selected_indices
                    ])
                else:
                    updates = self._ray.get([
                        self.clients[i].local_train.remote(budget) for i in selected_indices
                    ])
            else:
                updates = self._ray.get([
                    self.clients[i].local_train_round.remote(ref, budget)
                    for i, ref in zip(selected_indices, per_client_refs, strict=True)
                ])

            for update in updates:
                self.weights_by_client[update.client_id] = update.weights
            if aggregation == "fedavg":
                self.global_weights = fedavg_aggregate(updates)

            summary = self._make_summary(round_idx, updates)
            if probe_steps:
                summary["total_env_steps"] += probe_steps
            if eval_every is not None and round_idx % eval_every == 0:
                summary["eval_returns"] = self._evaluate_all(eval_n_episode)
            if eval_pre_post:
                summary["eval_pre"] = pre
                summary["eval_post"] = post
                if post_k is not None:
                    summary["eval_post_k"] = post_k
            history.append(summary)
            if on_round is not None:
                on_round(summary)
            if checkpoint_dir is not None:
                client_states = self._ray.get([
                    client.get_full_state.remote() for client in self.clients
                ])
                save_checkpoint(
                    checkpoint_dir,
                    manifest={"next_round": round_idx + 1},
                    global_weights=self.global_weights,
                    client_states=client_states,
                )
            if verbose:
                self._print_summary(summary)
        return history

    def _sample_client_indices(
        self,
        clients_per_round: int | None,
        rng: random.Random,
    ) -> list[int]:
        if clients_per_round is None or clients_per_round >= len(self.clients):
            return list(range(len(self.clients)))
        if clients_per_round <= 0:
            raise ValueError("clients_per_round must be positive")
        return rng.sample(range(len(self.clients)), clients_per_round)

    def _make_summary(self, round_idx: int, updates: list[ClientUpdate]) -> dict[str, Any]:
        train_returns = [u.mean_return for u in updates if u.mean_return is not None]
        return {
            "round": round_idx,
            "n_clients": len(updates),
            "total_env_steps": sum(u.num_steps for u in updates),
            # NOTE: training-collector returns are biased towards short
            # episodes; use eval_returns (when eval_every is set) for learning
            # curves.
            "train_mean_return": float(sum(train_returns) / len(train_returns))
            if train_returns
            else None,
        }

    def _evaluate_all(self, n_episode: int) -> dict[int, float]:
        return self._evaluate_selected(range(len(self.clients)), n_episode)

    def _evaluate_selected(
        self,
        selected_indices: list[int] | range,
        n_episode: int,
    ) -> dict[int, float]:
        eval_futures = [
            self.clients[i].evaluate.remote(n_episode=n_episode) for i in selected_indices
        ]
        evals = self._ray.get(eval_futures)
        return {
            self.client_ids[i]: float(result["mean_return"])
            for i, result in zip(selected_indices, evals, strict=True)
        }

    def _print_summary(self, summary: dict[str, Any]) -> None:
        line = (
            f"[round {summary['round']:03d}] clients={summary['n_clients']} "
            f"steps={summary['total_env_steps']} "
        )
        if summary.get("eval_returns"):
            returns = list(summary["eval_returns"].values())
            mean = float(sum(returns) / len(returns))
            line += f"eval_mean={mean:.1f} eval_per_client=" + "{" + ", ".join(
                f"{cid}:{v:.0f}" for cid, v in summary["eval_returns"].items()
            ) + "}"
        else:
            tmr = summary["train_mean_return"]
            line += f"train_mean_return={tmr:.2f}" if tmr is not None else "train_mean_return=None"
        if summary.get("eval_pre") is not None:
            drops = {
                cid: summary["eval_post"][cid] - summary["eval_pre"][cid]
                for cid in summary["eval_pre"]
            }
            line += " sync_drop=" + "{" + ", ".join(
                f"{cid}:{v:+.1f}" for cid, v in drops.items()
            ) + "}"
        print(line)
