"""End-to-end tests for the single-port cross-container bridge.

Runs a mini local Ray cluster in-process, serves a group of client actors
through a FederatedWorkerServer on a loopback port, and drives it through
RemoteClientGroup — including a full federated round mixing a local actor
with a remote group (the exact pattern the coordinator must support).
"""
from __future__ import annotations

import threading

import pytest

from examples.federated.algorithm_builders import build_dqn_cartpole, make_cartpole_env_fn
from tianshou.federated.adapter import OffPolicyFederatedAdapter
from tianshou.federated.client_actor import remote_client_actor
from tianshou.federated.coordinator import FederatedCoordinator
from tianshou.federated.remote_bridge import FederatedWorkerServer, RemoteClientGroup
from tianshou.federated.weight_utils import LocalBudget


@pytest.fixture(scope="module")
def ray_context():
    import ray

    ray.init(num_cpus=4, ignore_reinit_error=True)
    yield ray
    ray.shutdown()


def build_actor(client_id: int):
    actor_cls = remote_client_actor(num_cpus=1)
    return actor_cls.remote(
        client_id=client_id,
        env_fn=make_cartpole_env_fn(9.8),
        algo_builder=build_dqn_cartpole,
        adapter_factory=OffPolicyFederatedAdapter,
        num_train_envs=2,
        buffer_size=1000,
        device="cpu",
        seed=42,
    )


def start_server(actors, ray):
    server = FederatedWorkerServer(actors, host="127.0.0.1", port=0)
    port = server.listener.address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port


def test_bridge_roundtrip(ray_context) -> None:
    ray = ray_context
    actors = [build_actor(i) for i in range(2)]
    server, port = start_server(actors, ray)
    try:
        groups = [RemoteClientGroup(client_id=i, host="127.0.0.1", port=port)
                  for i in range(2)]
        weights = ray.get(groups[0].get_initial_weights.remote())
        assert "_optimizers" not in weights
        assert any(name.startswith("policy.") for name in weights)

        # sync + train + evaluate through the bridge
        ray.get(groups[0].sync_weights.remote(ray.put(weights)))
        ray.get(groups[1].sync_weights.remote(ray.put(weights)))
        updates = ray.get(groups[0].local_train.remote(LocalBudget(50, 5, 32)))
        assert updates.num_steps > 0

        evals = ray.get(groups[1].evaluate.remote(n_episode=2))
        assert "mean_return" in evals

        info = ray.get(groups[1].get_device_info.remote())
        assert "device" in info
    finally:
        server.stop()


def test_bridge_mixed_coordinator_round(ray_context) -> None:
    """A full FedAvg round mixing one local actor and one remote group."""
    ray = ray_context
    local_actor = build_actor(0)
    remote_actors = [build_actor(1)]
    server, port = start_server(remote_actors, ray)
    try:
        group = RemoteClientGroup(client_id=1, host="127.0.0.1", port=port)
        coordinator = FederatedCoordinator([local_actor, group])
        history = coordinator.run(
            num_rounds=2,
            budget=LocalBudget(50, 5, 32),
            eval_every=2,
            eval_n_episode=2,
            verbose=False,
        )
        assert len(history) == 2
        assert history[-1]["n_clients"] == 2
        assert len(history[-1]["eval_returns"]) == 2
    finally:
        server.stop()


def test_bridge_error_propagation(ray_context) -> None:
    ray = ray_context
    actors = [build_actor(0)]
    server, port = start_server(actors, ray)
    try:
        group = RemoteClientGroup(client_id=0, host="127.0.0.1", port=port)
        with pytest.raises(RuntimeError, match="unknown bridge command"):
            group._call("no_such_cmd", ())
    finally:
        server.stop()
