"""ML-7 causal collector state and bounded asynchronous inference.

No fitting, imputation, label input, or modification of the ML-6 package.
Collector state is owned by one event loop; only immutable windows cross threads.
"""
from collections import deque
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import queue
import threading
import time

import numpy as np

ROUTERS = tuple(f"r{i}" for i in range(1, 11))
COUNTERS = ("nInInterests", "nOutInterests", "nInData", "nOutData",
            "nInNacks", "nOutNacks", "nSatisfiedInterests", "nUnsatisfiedInterests")
GAUGES = ("nPitEntries", "nCsEntries")
FEATURES = COUNTERS + GAUGES
ARRAYS = ("x", "mask", "age", "age_valid", "feature_valid")


@dataclass(frozen=True)
class Schedule:
    start_ns: int
    count: int = 50
    interval_ns: int = 3_000_000_000
    stagger_ns: int = 100_000_000
    deadline_ns: int = 1_900_000_000

    def __post_init__(self):
        if type(self.start_ns) is not int or self.start_ns < 0:
            raise ValueError("Invalid start")
        if type(self.count) is not int or not 1 <= self.count <= 100_000:
            raise ValueError("Invalid count")
        if (self.interval_ns, self.stagger_ns, self.deadline_ns) != (3_000_000_000, 100_000_000, 1_900_000_000):
            raise ValueError("Frozen cadence/deadline contract mismatch")

    def target(self, router, seq):
        if router not in ROUTERS or type(seq) is not int or not 1 <= seq <= self.count:
            raise ValueError("Unknown router/sequence")
        return self.start_ns + (seq - 1) * self.interval_ns + ROUTERS.index(router) * self.stagger_ns

    def deadline(self, router, seq):
        return self.target(router, seq) + self.deadline_ns

    def decision(self, seq):
        return self.deadline(ROUTERS[-1], seq)


class CausalCollector:
    def __init__(self, run_id, boot_id, schedule):
        if not isinstance(run_id, str) or not isinstance(boot_id, str) or not run_id or not boot_id:
            raise ValueError("Run and boot required")
        self.run_id, self.boot_id, self.schedule = run_id, boot_id, schedule
        self.pending, self.seen, self.previous = {}, set(), {}
        self.frames = deque(maxlen=5)
        self.closed_sequence = 0

    def receive(self, payload, received_ns, available_ns=None):
        # Select by whitelist; scenario metadata never enters tensors.
        p = json.loads(payload) if isinstance(payload, (str, bytes)) else dict(payload)
        if not isinstance(p, dict):
            raise ValueError("Report must be a JSON object")
        if p.get("run_id") != self.run_id or p.get("boot_id") != self.boot_id:
            raise ValueError("Run/boot mismatch")
        router, seq, created = p["router_id"], p["sequence"], p["created_ns"]
        target = self.schedule.target(router, seq)
        if type(created) is not int or type(received_ns) is not int or not target <= created <= received_ns:
            raise ValueError("Invalid timestamp ordering")
        metrics = p["metrics"]
        if set(metrics) != set(FEATURES) or any(type(v) is not int or v < 0 or v > 2**64 - 1 for v in metrics.values()):
            raise ValueError("Invalid metrics")
        instance = p["nfd_instance_id"]
        if not isinstance(instance, str) or not instance:
            raise ValueError("NFD instance required")
        key = (router, seq)
        if key in self.seen:
            raise ValueError("Duplicate report")
        if available_ns is not None and (type(available_ns) is not int or available_ns < received_ns):
            raise ValueError("Invalid sidecar availability timestamp")
        self.seen.add(key)
        if seq <= self.closed_sequence:
            return "closed_frame"
        if available_ns is not None:
            if available_ns > self.schedule.decision(seq):
                return "ipc_after_decision_excluded"
        if received_ns > self.schedule.deadline(router, seq):
            return "late_report_excluded"
        self.pending[key] = {"created_ns": created, "received_ns": received_ns,
                             "nfd_instance_id": instance, "metrics": dict(metrics)}
        return "on_time"

    def close(self, seq):
        if seq != self.closed_sequence + 1 or seq > self.schedule.count:
            raise ValueError("Frames must close once in sequence")
        decision = self.schedule.decision(seq)
        frame = {"x": np.zeros((10, 10), np.float32),
                 "feature_valid": np.zeros((10, 10), np.float32),
                 "mask": np.zeros(10, np.float32), "age": np.zeros(10, np.float32),
                 "age_valid": np.zeros(10, np.float32)}
        reasons = []
        for i, router in enumerate(ROUTERS):
            current = self.pending.pop((router, seq), None)
            prev = self.previous.get(router)
            if current is not None:
                frame["mask"][i] = 1
                frame["x"][i, 8:] = [current["metrics"][k] for k in GAUGES]
                frame["feature_valid"][i, 8:] = 1
                if prev is None:
                    reason = "first_observation"
                else:
                    elapsed = (current["created_ns"] - prev["created_ns"]) / 1e9
                    delta = [current["metrics"][k] - prev["metrics"][k] for k in COUNTERS]
                    reset = current["nfd_instance_id"] != prev["nfd_instance_id"] or elapsed <= 0 or min(delta) < 0
                    reason = "counter_or_instance_reset" if reset else "observed"
                    if not reset:
                        frame["x"][i, :8] = [v / elapsed for v in delta]
                        frame["feature_valid"][i, :8] = 1
                self.previous[router] = current
                prev = current
            else:
                reason = "missing_or_late_zero_features"
            if prev is not None:
                frame["age"][i] = (decision - prev["created_ns"]) / 1e9
                frame["age_valid"][i] = 1
            reasons.append(reason)
        if not all(np.isfinite(v).all() for v in frame.values()):
            raise ValueError("Nonfinite collector frame")
        self.closed_sequence = seq
        for v in frame.values():
            v.setflags(write=False)
        self.frames.append(frame)
        window = None
        if len(self.frames) == 5:
            window = {k: np.stack([f[k] for f in self.frames])[None] for k in ARRAYS}
            for v in window.values():
                v.setflags(write=False)
        return frame, window, reasons


def frozen_inventory(package):
    return {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(package).rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}


def load_runtime(package, model_id):
    """Load the *packaged* ML-6 entry point, after explicit integrity checks.

    Checks here remain active under python -O (ML-6 uses some assertions).
    """
    package = Path(package).resolve()
    registry = json.loads((package / "registry.json").read_text())
    expected = {"schema_version": 1, "threshold": 0.5, "router_order": list(ROUTERS),
                "window_steps": 5, "interval_seconds": 3, "node_count": 10,
                "feature_count": 10, "encoded_channels": 13}
    if any(registry.get(k) != v for k, v in expected.items()):
        raise ValueError("ML-6 registry contract mismatch")
    entries = [m for m in registry["models"] if m["id"] == model_id]
    if len(entries) != 1 or entries[0]["variant"] != "mask_age":
        raise ValueError("ML-7 requires an explicit registered Mask+Age model")
    entry = entries[0]
    hashes = dict(registry["runtime_sha256"])
    hashes.update({entry["checkpoint"]: entry["checkpoint_sha256"],
                   entry["encoder"]: entry["encoder_sha256"],
                   registry["edge_index_path"]: registry["edge_index_sha256"]})
    for relative, digest in hashes.items():
        path = (package / relative).resolve()
        if not path.is_relative_to(package) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"ML-6 integrity failure: {relative}")
    schema = json.loads((package / "assets/feature_schema.json").read_text())
    if schema["raw_features"] != list(FEATURES) or schema["router_order"] != list(ROUTERS):
        raise ValueError("Feature/router order mismatch")
    spec = importlib.util.spec_from_file_location("frozen_ml6_runtime", package / "scripts/ml6_workflow.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Import path isolation: fail rather than reuse a previously imported encoder
    # from a different checkout. Service is intended to run in its own process.
    import sys
    sys.dont_write_bytecode = True
    for name in ("ml3_preprocessing", "ml4_observation", "models.graphsage_gru"):
        if name in sys.modules:
            source = Path(sys.modules[name].__file__).resolve()
            if not source.is_relative_to(package):
                raise ValueError(f"Unfrozen imported module: {name}")
    runtime = mod.OfflineInference(package, model_id, "cpu")
    encoder = json.loads((package / entry["encoder"]).read_text())
    names = [f"{k}_rate" for k in COUNTERS] + list(GAUGES) + ["mask", "age_log1p_standardized", "age_valid"]
    if encoder["channel_names"] != names or encoder["train_runs"] != entry["train_runs"]:
        raise ValueError("Encoder provenance mismatch")
    return runtime, entry, hashlib.sha256((package / "registry.json").read_bytes()).hexdigest()


class AsyncInference:
    """One in-flight request + bounded waiting queue; caller never waits on model.

    A stuck model cannot grow the queue. The owner calls poll() to expire requests.
    Expired results are discarded; fallback is UNKNOWN, never cached Normal/IFA.
    """
    def __init__(self, runtime, sink, model_id, clock=time.monotonic_ns,
                 timeout_ms=500, capacity=1, provenance=None):
        if timeout_ms <= 0 or capacity < 1:
            raise ValueError("Invalid inference budget")
        self.runtime, self.sink, self.model_id, self.clock = runtime, sink, model_id, clock
        self.timeout_ns = int(timeout_ms * 1e6)
        self.jobs, self.results = queue.Queue(capacity), queue.Queue(capacity + 1)
        self.outstanding, self.next_id = {}, 0
        self.stop = threading.Event()
        self.provenance = provenance or {}
        self.thread = threading.Thread(target=self._worker, daemon=True, name="ml7-inference")
        self.thread.start()

    def emit(self, meta, reason, **extra):
        published = self.clock()
        acquisition = {}
        if meta.get("current_report_age_max_ms") is not None:
            acquisition["sample_end_to_publish_max_ms"] = meta["current_report_age_max_ms"] + (published-meta["decision_ns"])/1e6
        self.sink({**self.provenance, **meta, "event": "decision", "model_id": self.model_id,
                   "status": "unknown", "reason": reason, "probability_ifa": None,
                   "prediction": None, "published_ns": published, **acquisition, **extra})

    def submit(self, window, meta):
        self.poll()
        now = self.clock()
        if self.stop.is_set():
            self.emit(meta, "shutdown")
            return False
        if now > meta["decision_ns"] + self.timeout_ns:
            self.emit(meta, "decision_expired")
            return False
        if window is None:
            self.emit(meta, "window_warmup")
            return False
        if not np.any(window["mask"][0, -1]):
            self.emit(meta, "no_current_reports")
            return False
        if len(self.outstanding) >= self.jobs.maxsize + 1:
            self.emit(meta, "queue_full")
            return False
        self.next_id += 1
        token = self.next_id
        job = (token, {k: v.copy() for k, v in window.items()}, dict(meta), now)
        try:
            self.jobs.put_nowait(job)
        except queue.Full:
            self.emit(meta, "queue_full")
            return False
        self.outstanding[token] = (dict(meta), now)
        return True

    def _worker(self):
        while not self.stop.is_set():
            try:
                token, arrays, meta, enqueued = self.jobs.get(timeout=.05)
            except queue.Empty:
                continue
            started = self.clock()
            error, probability, prediction = None, None, None
            if started > meta["decision_ns"] + self.timeout_ns:
                error = "inference_timeout"
            else:
                try:
                    probabilities, predictions = self.runtime.predict(arrays)
                    probability, prediction = float(probabilities[0]), int(predictions[0])
                    if not math.isfinite(probability) or not 0 <= probability <= 1 or prediction != int(probability >= .5):
                        raise ValueError("Invalid model output")
                except Exception as exc:
                    error = f"runtime_error:{type(exc).__name__}"
            finished = self.clock()
            result = (token, probability, prediction, error, started, finished)
            # Queue bounded by admitted requests; expired completion queue may
            # still be full. Dropping completion cannot publish a stale decision.
            try:
                self.results.put_nowait(result)
            except queue.Full:
                pass

    def poll(self):
        now = self.clock()
        while True:
            try:
                token, probability, prediction, error, started, finished = self.results.get_nowait()
            except queue.Empty:
                break
            request = self.outstanding.pop(token, None)
            if request is None:
                continue
            meta, enqueued = request
            timing = {"queue_wait_ms": (started-enqueued)/1e6,
                      "inference_ms": (finished-started)/1e6,
                      "decision_to_publish_ms": (now-meta["decision_ns"])/1e6}
            if now > meta["decision_ns"] + self.timeout_ns:
                self.emit(meta, "inference_timeout", **timing)
            elif error:
                self.emit(meta, error, **timing)
            else:
                self.emit(meta, "model_output", status="prediction", probability_ifa=probability,
                          prediction=prediction, **timing)
        for token, (meta, _) in list(self.outstanding.items()):
            if now > meta["decision_ns"] + self.timeout_ns:
                del self.outstanding[token]
                self.emit(meta, "inference_timeout")

    def close(self):
        self.poll()
        self.stop.set()
        for meta, _ in self.outstanding.values():
            self.emit(meta, "shutdown")
        self.outstanding.clear()
        self.thread.join(timeout=.1)


class JsonlAudit:
    """Nonblocking audit enqueue; logging failure is fatal to the owner loop."""
    def __init__(self, path, capacity=4096):
        self.file = open(path, "x", encoding="utf-8")
        self.queue = queue.Queue(capacity)
        self.error = None
        self.thread = threading.Thread(target=self._write, daemon=True, name="ml7-audit")
        self.thread.start()

    def __call__(self, event):
        self.check()
        try:
            self.queue.put_nowait(dict(event))
        except queue.Full as exc:
            self.error = RuntimeError("Audit queue full; stop inference service")
            raise self.error from exc

    def check(self):
        if self.error:
            raise RuntimeError("Audit writer failed") from self.error

    def _write(self):
        try:
            while True:
                event = self.queue.get()
                if event is None:
                    break
                self.file.write(json.dumps(event, allow_nan=False) + "\n")
                self.file.flush()
        except Exception as exc:
            self.error = exc
        finally:
            self.file.close()

    def close(self):
        self.check()
        self.queue.put(None, timeout=1)
        self.thread.join(timeout=2)
        self.check()
        if self.thread.is_alive():
            raise RuntimeError("Audit drain timeout")
