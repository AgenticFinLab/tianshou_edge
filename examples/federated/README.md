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

## Implementation status

Implemented and validated: DQN / SAC / PPO federated training via one config
file, FedAvg and actor-scope FedProx, partial participation, pre/post
aggregation diagnostics, checkpoint/resume (full state or model-only),
streaming per-client results (CSV + history JSONL), per-actor GPU placement
reporting. Validated on a single-node 8xA40 container: see
`tianshou/federated/README.md` for the validation summary and
`results/benchmark_server.md` for the three-mode benchmark.

Not yet implemented: MARL (per-agent aggregation), secure aggregation /
differential privacy, multi-node Ray clusters (single-node multi-GPU only,
so far).

## Multi-node deployment

Two supported modes:

### A. Single-port bridge (recommended for isolated networks)

When nodes cannot reach each other on Ray's ports (isolated containers, NAT),
use the application-level bridge: every node keeps its own LOCAL Ray cluster
(fully stable) and nodes exchange weights through one authenticated TCP port.

Architecture:

* Remote node: run `python examples/federated/run_remote_server.py
  --config <remote config> --port 18888`. This builds the remote clients and
  serves them through `FederatedWorkerServer` (127.0.0.1:18888).
* Head node: forward exactly one port with an SSH tunnel
  (`ssh -N -L 0.0.0.0:18888:127.0.0.1:18888 <remote-host>`), then wrap each
  remote client in `tianshou.federated.remote_bridge.RemoteClientGroup` and
  pass it to the coordinator alongside local actors — the coordinator runs
  unchanged (the group mimics a Ray actor handle).
* Tensors are encoded to bytes explicitly (`_encode`/`_decode` in
  `remote_bridge.py`): torch's default pickler shares storages through
  unix-socket file descriptors, which cannot cross containers.

Verified across two isolated containers (acceptance: 100-round mixed
federated sync, 1 local A40 + 4 remote A800-MIG, 3.74s/round; results in
`results/multinode_acceptance.md`, kept local).

Remote config note: some containers expose only a few CPUs to Ray
(`ray.cluster_resources()` reports e.g. 4.0). Size `num_cpus` per actor so
that `clients_count x num_cpus` fits below the detected CPU count, otherwise
actor construction stalls silently (see
`config_examples/remote_halfcheetah.yaml`, `num_cpus: 0.5`).

### B. Native Ray cluster (when ports are reachable)

The federated layer is Ray-native: a multi-node run only needs a Ray cluster
spanning the nodes and the repository available at the same path on every
node. No code changes are required — the unified entry joins the cluster via
`ray.address` in the config.

Requirements per node:

* the repository at the same absolute path (or `pip install -e .` plus the
  `examples` package on every node),
* the same conda env with matching `torch`/`ray` versions,
* network reachability plus open Ray ports (GCS 6379, dashboard 8265, and
  the worker port range around 10001-10999),
* GPUs visible to the container (`nvidia-smi`).

Cluster startup (one node becomes the head):

```bash
# on the head node
ray start --head --port=6379 --dashboard-host 0.0.0.0

# on every worker node
ray start --address=<head-ip>:6379
```

Run the federated experiment from the head (driver side); set the address in
the config:

```yaml
ray:
  address: "<head-ip>:6379"
  num_cpus: 8
  num_gpus: 0.25
```

Verification: the entry prints `ray cluster nodes: N` at startup and every
actor reports `node_id`/`node_ip` in its `get_device_info` output — placement
is verified per actor, per node, instead of trusting resource declarations.
Results are written by the driver and land on the head node's local
`results/`; for shared access mount the same volume on all nodes.

## Where the pieces live

* `tianshou/federated/` — the platform: coordinator, client actor, adapters,
  weight utilities, persistence, config schema (API contract in
  `tianshou/federated/README.md`).
* `examples/federated/` — unified entry, config examples, quickstart scripts, algorithm builders.
* `test/federated/` — unit tests (`pytest test/federated`).
