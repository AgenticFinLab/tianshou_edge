"""Federated reinforcement learning layer built on Ray Core and tianshou.

Clients train locally on their own environments and replay buffers; a
driver-side coordinator aggregates shareable model weights (FedAvg) and
broadcasts them each round. Raw transitions never leave a client.

Design:

* Driver-driven control flow: the :class:`FederatedCoordinator` explicitly
  triggers each client round, so partial participation and straggler handling
  stay possible at the driver level. Clients never block inside trainer
  callbacks.
* Per-algorithm adapters (:class:`FederatedAlgorithmAdapter`) declare which
  parameters are shared, how target networks are treated, and how the local
  training loop runs.
* Weights are shipped through Ray's object store as plain CPU
  ``dict[str, torch.Tensor]`` payloads; optimizer state stays client-local.

Ray-dependent modules (:mod:`tianshou.federated.client_actor`,
:mod:`tianshou.federated.coordinator`) are intentionally not imported here, so
``import tianshou.federated`` works without Ray installed.

Example (see ``examples/federated/run_dqn_smoke.py``)::

    actor_cls = remote_client_actor(num_cpus=1)
    clients = [actor_cls.remote(client_id=i, env_fn=..., algo_builder=..., adapter_factory=...) for i in range(3)]
    coordinator = FederatedCoordinator(clients)
    history = coordinator.run(num_rounds=50, budget=LocalBudget(...))
"""
from tianshou.federated.adapter import (
    FederatedAlgorithmAdapter,
    OffPolicyFederatedAdapter,
    SACFederatedAdapter,
)
from tianshou.federated.weight_utils import (
    ClientUpdate,
    FedProxGradientHook,
    LocalBudget,
    LocalTrainStats,
    extract_fed_state_dict,
    fedavg_aggregate,
    load_fed_state_dict,
)

__all__ = [
    "FederatedAlgorithmAdapter",
    "OffPolicyFederatedAdapter",
    "SACFederatedAdapter",
    "ClientUpdate",
    "FedProxGradientHook",
    "LocalBudget",
    "LocalTrainStats",
    "extract_fed_state_dict",
    "fedavg_aggregate",
    "load_fed_state_dict",
]
