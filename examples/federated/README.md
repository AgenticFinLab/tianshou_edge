# Federated RL quickstart (DQN / SAC / PPO)

Run multi-client federated experiments on tianshou with a single config
file. This tutorial covers installation, the three algorithm quickstarts,
reading results, and resuming interrupted runs.

## Requirements

* Python 3.11
* `ray` and `pyyaml` (installed via the `federated` extra, or manually)
* tianshou installed from this repository (editable):

```bash
conda create -n tianshou_fed python=3.11 -y
conda activate tianshou_fed
pip install -e .[federated]
# or, without the extra:
# pip install -e . && pip install ray pyyaml
```

## Quickstart

From the repository root, launch one of the ready-made configs
(`examples/federated/config_examples/`):

```bash
# 1. Federated DQN, 3 CartPole clients with different gravity
python examples/federated/run_federated.py \
    --config examples/federated/config_examples/dqn_cartpole.yaml

# 2. Federated SAC, 3 Pendulum clients (calibrated config, actor-only sharing)
python examples/federated/run_federated.py \
    --config examples/federated/config_examples/sac_pendulum.yaml

# 3. Federated PPO, 3 CartPole clients (on-policy rounds)
python examples/federated/run_federated.py \
    --config examples/federated/config_examples/ppo_cartpole.yaml
```

Each command starts a local Ray cluster, trains 3 clients in parallel, and
prints a per-round progress line plus a final per-client table.

## Reading results

* **CSV** (`output.csv_path`): one row per client evaluation —
  `algorithm, seed, round, client_id, eval_return, aggregation, agg_scope`.
  Rows are streamed (flushed) as they are produced, so an interrupted run
  keeps everything up to the last completed round.
* **History JSONL** (`<csv stem>_history.jsonl`): the full per-round summary
  (steps, train/eval returns per client, and `sync_drop` when
  `evaluation.pre_post_eval` is enabled).
* The config path and a summary of its key fields are echoed to stdout at
  start, making every run reproducible from its log.

## Checkpointing and resume

Set `output.checkpoint_dir` in the config to write a checkpoint after every
round (`manifest.json` + aggregated weights + full client states). Resume
from any checkpoint:

```bash
# exact continuation: weights + optimizer momentum + replay buffers + RNG
python examples/federated/run_federated.py --config my_config.yaml \
    --resume <checkpoint_dir>

# model-only continuation: restore the aggregated weights, fresh buffers
python examples/federated/run_federated.py --config my_config.yaml \
    --resume <checkpoint_dir> --resume-mode model_only
```

Rounds are counted from the checkpoint's `next_round`; results are appended
to the same CSV (no overwrite).

## Config reference

See `tianshou/federated/config.py` for the full schema. The key sections:

```yaml
algorithm: dqn | sac | ppo        # which algorithm family
env: cartpole | pendulum          # which environment
seeds: [42, 43]                   # one run per seed
device: cpu | cuda
clients_count: 3                  # number of federated clients
envs_per_client: 4                # parallel envs per client
buffer_size: 10000
warmup_steps: 0                   # random-action warm-up per client
aggregation: fedavg | local_only  # local_only = no-communication baseline
agg_scope: all | actor_only       # SAC/PPO: share actor only or all modules
fedprox_mu: 0.0                   # > 0 enables actor-scope FedProx
heterogeneity:                    # per-client physics; scalar broadcasts
  gravity: [9.3, 9.8, 10.3]       #   or a list with one entry per client
  mass: 1.0
  length: 1.0
algorithm_config:                 # hidden_sizes, lr, per-algorithm hyperparameters
budget:
  rounds: 20
  collect_per_update: 100         # off-policy: steps before each update;
  updates_per_round: 10           #   on-policy: steps and epochs per round
  batch_size: 64
evaluation:
  every_rounds: 2
  episodes: 10
  pre_post_eval: false            # record sync_drop per round (doubles eval traffic)
ray:
  num_cpus: 1                     # resources per client actor
  num_gpus: 0
output:
  csv_path: results/my_run.csv
  checkpoint_dir: null            # set a path to checkpoint every round
```

## Not yet supported

* MARL (per-agent aggregation) — the adapter interface is designed to
  accommodate it, but no implementation exists yet.
* Secure aggregation / differential privacy.
* Multi-node Ray clusters (untested; single-node is the supported mode).

## Where the pieces live

* `tianshou/federated/` — the platform: coordinator, client actor, adapters,
  weight utilities, persistence, config schema (API contract in
  `tianshou/federated/README.md`).
* `examples/federated/` — unified entry, config examples, standalone smoke
  scripts, algorithm builders.
* `test/federated/` — unit tests (`pytest test/federated`).
