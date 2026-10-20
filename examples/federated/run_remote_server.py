"""Cross-container bridge server entry: serves one container's clients over one port.

Runs a LOCAL Ray cluster (never cross-node Ray), builds the clients from the
same config schema as ``run_federated.py``, and serves them through
:class:`tianshou.federated.remote_bridge.FederatedWorkerServer`. The head
container reaches this port through a single SSH tunnel and drives the
clients via ``RemoteClientGroup``.

Usage on the remote container::

    python examples/federated/run_remote_server.py \
        --config examples/federated/config_examples/remote_halfcheetah.yaml \
        --port 18888
"""
from __future__ import annotations

import argparse

from examples.federated.run_federated import build_clients
from tianshou.federated.config import load_config
from tianshou.federated.remote_bridge import FederatedWorkerServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True,
                        help="config for THIS container's clients (.yaml or .json)")
    parser.add_argument("--port", type=int, default=18888,
                        help="TCP port to serve on (single-port bridge)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    import ray  # lazy import

    ray.init(ignore_reinit_error=True)  # LOCAL cluster, never cross-node
    clients = build_clients(cfg, seed=args.seed)
    server = FederatedWorkerServer(clients, host="127.0.0.1", port=args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
