"""Cross-container federated bridge: local Ray per container, one TCP port between.

Multi-node Ray clusters need bidirectional connectivity on many ports, which
isolated containers cannot provide reliably (Ray's raylet also proved
fragile over NAT tunnels). This bridge keeps Ray strictly local to each
container — where it is stable — and connects containers with a single
authenticated TCP stream:

* Each container runs its own local Ray cluster and its own
  ``TianshouFedClientActor`` actors (full stability, zero cross-node Ray).
* The worker side runs a :class:`FederatedWorkerServer` listening on one
  port (e.g. 127.0.0.1:18888, reachable from the head via a single SSH
  tunnel). It executes commands against its local actors in parallel via
  ``ray.get`` and replies over the same connection.
* The head side wraps each remote client in :class:`RemoteClientGroup`, which
  mimics a Ray actor handle (attribute access returns a method proxy with a
  ``.remote()`` call). Results are returned as local ``ray.ObjectRef``s via
  ``ray.put``, so the existing :class:`~tianshou.federated.coordinator.
  FederatedCoordinator` can drive a MIX of local actors and remote groups
  without any change (the coordinator assumes one client per handle).

Security note: the pickle-based stream is authenticated with a shared
``authkey``; only bind to interfaces reachable by trusted peers.
"""
from __future__ import annotations

import dataclasses
import importlib
import threading
from multiprocessing.connection import Client, Listener
from typing import Any

import numpy as np
import torch

__all__ = ["FederatedWorkerServer", "RemoteClientGroup"]

AUTHKEY = b"tianshou-federated-bridge"


def _encode(obj: Any) -> Any:
    """Convert torch tensors and numpy arrays to fully serializable bytes.

    PyTorch's default pickler shares storages through unix-socket file
    descriptors (resource_sharer), which only works on one machine; the
    bridge must carry payloads across containers, so tensors are converted
    to plain (bytes, dtype, shape) tuples instead.
    """
    if isinstance(obj, torch.Tensor):
        t = obj.detach().cpu()
        return {"__torch_tensor__": (t.numpy().tobytes(), str(t.numpy().dtype), tuple(t.shape))}
    if isinstance(obj, np.ndarray):
        return {"__numpy__": (obj.tobytes(), str(obj.dtype), tuple(obj.shape))}
    if dataclasses.is_dataclass(obj):
        return {
            "__dataclass__": f"{type(obj).__module__}.{type(obj).__qualname__}",
            "fields": _encode(dataclasses.asdict(obj)),
        }
    if isinstance(obj, dict):
        return {key: _encode(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(value) for value in obj]
    return obj


def _decode(obj: Any) -> Any:
    if isinstance(obj, dict) and "__torch_tensor__" in obj:
        data, dtype, shape = obj["__torch_tensor__"]
        return torch.from_numpy(np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape))
    if isinstance(obj, dict) and "__numpy__" in obj:
        data, dtype, shape = obj["__numpy__"]
        return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape)
    if isinstance(obj, dict) and "__dataclass__" in obj:
        module, name = obj["__dataclass__"].rsplit(".", 1)
        cls = getattr(importlib.import_module(module), name)
        return cls(**_decode(obj["fields"]))
    if isinstance(obj, dict):
        return {key: _decode(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_decode(value) for value in obj]
    return obj


def _decode_remote_args(args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """Resolve Ray ObjectRefs inside arguments before shipping them over TCP."""
    import ray

    def convert(value: Any) -> Any:
        if isinstance(value, ray.ObjectRef):
            return ray.get(value)
        return value

    return tuple(convert(a) for a in args), {k: convert(v) for k, v in kwargs.items()}


class FederatedWorkerServer:
    """Serves commands against this container's local Ray actors over one TCP port.

    :param clients: local Ray actor handles of
        :class:`~tianshou.federated.client_actor.TianshouFedClientActor`.
    :param host: bind address (default 127.0.0.1; reachable via SSH tunnel
        from the head container).
    :param port: listen port.
    """

    def __init__(self, clients: list[Any], host: str = "127.0.0.1", port: int = 18888) -> None:
        import ray

        self.clients = clients
        ids = ray.get([c.get_client_id.remote() for c in clients])
        self._by_id = dict(zip(ids, clients, strict=True))
        self.listener = Listener((host, port), authkey=AUTHKEY)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        print(f"[bridge] FederatedWorkerServer listening on {host}:{self.listener.address[1]} "
              f"with {len(clients)} clients (ids {ids})", flush=True)

    def serve_forever(self) -> None:
        """Block the calling thread, accepting and handling connections."""
        self._thread = threading.current_thread()
        while not self._stop.is_set():
            try:
                conn = self.listener.accept()
            except OSError:
                break  # listener closed
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        self.listener.close()

    def _actor(self, client_id: int):
        actor = self._by_id.get(client_id)
        if actor is None:
            raise ValueError(f"unknown client_id {client_id!r}; known ids: {sorted(self._by_id)}")
        return actor

    def _handle(self, conn) -> None:
        try:
            while True:
                msg = _decode(conn.recv())
                cmd = msg["cmd"]
                args, kwargs = _decode_remote_args(msg.get("args", ()), msg.get("kwargs", {}))
                try:
                    result = self._dispatch(cmd, args, kwargs)
                    conn.send(_encode({"ok": True, "result": result}))
                except Exception as e:  # noqa: BLE001 - report errors to the caller
                    conn.send(_encode({"ok": False, "error": f"{type(e).__name__}: {e}"}))
        except (EOFError, ConnectionResetError, OSError):
            pass  # head disconnected
        finally:
            conn.close()

    def _dispatch(self, cmd: str, args: tuple, kwargs: dict) -> Any:
        import ray

        if cmd == "get_client_id":
            (client_id,) = args
            return client_id
        if cmd == "get_initial_weights":
            (client_id,) = args
            return ray.get(self._actor(client_id).get_initial_weights.remote())
        if cmd == "sync_weights":
            client_id, weights = args
            return ray.get(self._actor(client_id).sync_weights.remote(weights))
        if cmd == "local_train_round":
            client_id, weights, budget = args
            return ray.get(self._actor(client_id).local_train_round.remote(weights, budget))
        if cmd == "local_train":
            client_id, budget = args
            return ray.get(self._actor(client_id).local_train.remote(budget))
        if cmd == "evaluate":
            (client_id,) = args
            return ray.get(self._actor(client_id).evaluate.remote(
                n_episode=kwargs.get("n_episode", 5)))
        if cmd == "get_device_info":
            (client_id,) = args
            return ray.get(self._actor(client_id).get_device_info.remote())
        if cmd == "get_full_state":
            (client_id,) = args
            return ray.get(self._actor(client_id).get_full_state.remote())
        if cmd == "load_full_state":
            client_id, state, mode = args
            return ray.get(self._actor(client_id).load_full_state.remote(state, mode=mode))
        raise ValueError(f"unknown bridge command {cmd!r}")


class RemoteClientGroup:
    """Mimics a Ray actor handle for ONE remote client.

    ``group.sync_weights.remote(weights)`` behaves like a Ray remote call but
    executes synchronously over the bridge connection; the result is returned
    as a local ``ray.ObjectRef`` so coordinator code that calls ``ray.get`` on
    a list of futures works unchanged.
    """

    def __init__(
        self,
        client_id: int,
        host: str = "127.0.0.1",
        port: int = 18888,
        timeout: float = 3600.0,
    ) -> None:
        self.client_id = client_id
        self.host = host
        self.port = port
        self.timeout = timeout
        self._lock = threading.Lock()  # one request in flight per client

    def _call(self, cmd: str, args: tuple, kwargs: dict | None = None) -> Any:
        with self._lock:
            conn = Client((self.host, self.port), authkey=AUTHKEY)
            try:
                conn.send(_encode({"cmd": cmd, "args": (self.client_id, *args),
                                   "kwargs": kwargs or {}}))
                reply = _decode(conn.recv())
            finally:
                conn.close()
        if not reply.get("ok"):
            raise RuntimeError(f"remote bridge error for {cmd!r}: {reply.get('error')}")
        return reply["result"]

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)

        class _MethodProxy:
            """Returned by attribute access; mirrors Ray's RemoteFunction."""

            def remote(_self, *args: Any, **kwargs: Any) -> Any:
                import ray

                # ObjectRefs in arguments must be resolved before shipping.
                args, kwargs = _decode_remote_args(args, kwargs)
                result = self._call(name, args, kwargs)
                return ray.put(result)

        return _MethodProxy()
