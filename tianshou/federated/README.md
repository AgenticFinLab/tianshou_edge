# tianshou.federated — API contract and state boundary

Quickstart and installation: see `examples/federated/README.md`.

Driver-driven federated RL on Ray Core. Clients train locally on their own
environments and buffers; a coordinator aggregates shareable weights (FedAvg)
and broadcasts them each round. Raw transitions never leave a client.

## Architecture

```
coordinator (driver) --broadcast--> client actors --upload--> coordinator
    FederatedCoordinator            TianshouFedClientActor (Ray actor)
    run(rounds, budget, ...)        envs + algorithm + buffer + collector
```

## Adapter interface (stable contract)

Every algorithm participates through a `FederatedAlgorithmAdapter`:

| Member | Meaning |
|---|---|
| `get_shared_state() -> dict[str, Tensor]` | CPU payload of shareable weights (never contains optimizer state — `nn.Module.state_dict` bypasses the `_optimizers` key) |
| `load_shared_state(state) -> (missing, unexpected)` | Loads a payload into the algorithm; tolerant of missing keys (partial aggregation) |
| `shared_keys` | Parameter-name prefixes that are shared (`None` = all); e.g. `["policy.actor."]` for actor-only sharing |
| `exclude_prefixes` | Prefixes that never enter the payload (reward-scale-dependent state, e.g. SAC's `alpha.`, PPO's `ret_rms.`) |
| `exclude_target_net` | Whether target networks are excluded from the payload (off-policy adapters exclude them and re-sync locally; soft-update algorithms never hard-reset) |
| `prox_prefixes` | Prefixes the FedProx proximal term applies to (value networks are never pulled; their TD targets are non-stationary) |
| `local_train(collector, buffer, budget)` | One round of local training (off-policy: collect+update interleaved, buffer persists; on-policy: collect -> epochs -> buffer reset) |

`LocalBudget`: `num_env_steps_per_update` x `num_updates` x `batch_size`.
Off-policy: steps collected before each update x number of updates (buffer
persists across rounds). On-policy: steps collected per round x local epochs
over that round's data (buffer cleared after each round). The parallel env
count multiplies the collected steps of each `collect(n_step=...)` call.

## State boundary

| State | Shared? | Notes |
|---|---|---|
| `policy.actor.*` | always | the policy being federated |
| `critic.*`, `critic2.*` (SAC/PPO) | `agg_scope="all"` only | client-private under `actor_only` |
| target networks (`*_old.*`) | per adapter | off-policy: excluded from payload, re-synced locally (hard for DQN — its native update; never hard-reset for soft-update SAC) |
| `alpha.*` (SAC), `ret_rms.*` (PPO) | never | reward-scale-dependent statistics stay local |
| optimizers (`_optimizers`) | never | momentum stays client-local |
| replay buffer, collector, envs | never | transitions never leave the client |
| eps / exploration noise / RNG | never | process-local scalars and streams |

## Coordination

* `FederatedCoordinator.run(num_rounds, budget, aggregation="fedavg"|"local_only",
  eval_every, eval_n_episode, clients_per_round, on_round=callback)`
* `aggregation="fedavg"` averages uploads weighted by collected env steps;
  `"local_only"` runs the same driver-driven loop with identity broadcast
  (fair-budget no-communication baseline).
* `on_round(summary)` streams per-round summaries (including per-client
  evaluation returns) to disk, so an interrupted run loses at most one round.
* Evaluation uses a dedicated test collector, runs outside the training-step
  context (deterministic policy), and records per-client returns.

## Supported / not supported

Supported: DQN family, SAC (via `OffPolicyFederatedAdapter` /
`SACFederatedAdapter`), PPO/A2C (via `OnPolicyFederatedAdapter`), FedAvg,
FedProx (actor-scope), partial participation (`clients_per_round`),
pre/post-aggregation evaluation diagnostics (`eval_pre_post`, off by
default), per-round checkpointing and resume (full state or model-only, see
`tianshou.federated.persistence`), config-driven runs
(`tianshou.federated.config`).

Not yet supported (interface kept extensible): MARL per-agent aggregation,
secure aggregation / DP. Multi-node Ray setups are untested.
