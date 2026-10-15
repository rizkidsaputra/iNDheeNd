#!/usr/bin/env python3
"""Collector M sidecar. Receives nonblocking AF_UNIX datagrams from v8 adapter.

Runs on the SAME Linux boot/monotonic clock as the collector/reporter. No TCP
timestamp conversion, source telemetry, labels, or training data are accepted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import selectors
import socket
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ml7_online import AsyncInference, CausalCollector, JsonlAudit, Schedule, load_runtime


class Service:
    def __init__(self, collector, inference, audit, clock=time.monotonic_ns):
        self.collector, self.inference, self.audit, self.clock = collector, inference, audit, clock
        self.next_seq = 1

    def ingest(self, event, observed_ns=None):
        observed_ns = self.clock() if observed_ns is None else observed_ns
        if not isinstance(event, dict):
            raise ValueError("Transport event must be a JSON object")
        # Drain reports before closing the current due frame. An event dispatched
        # after closure is rejected even if its producer timestamp was on time.
        if event.get("event") != "telemetry_received":
            return
        received = event["monotonic_ns"]
        if type(received) is not int or received > observed_ns:
            raise ValueError("Future transport timestamp")
        p = json.loads(event["raw_payload"])
        router, seq = p["router_id"], p["sequence"]
        name = f"/monitor/{router}/telemetry/poc11c/{self.collector.run_id}/{seq}"
        if event.get("data_name") != name or event.get("router_id") != router or event.get("sequence") != seq:
            raise ValueError("NDN name/payload binding mismatch")
        result = self.collector.receive(p, received, available_ns=observed_ns)
        self.audit({"event": "report", "run_id": self.collector.run_id,
                    "router_id": router, "sequence": seq, "received_ns": received,
                    "created_ns": p["created_ns"],
                    "sample_end_to_receive_ms": (received-p["created_ns"])/1e6,
                    "ipc_observed_ns": observed_ns, "disposition": result,
                    "ipc_delay_ms": (observed_ns-received)/1e6,
                    "raw_sha256": hashlib.sha256(event["raw_payload"].encode()).hexdigest()})

    def tick(self, now=None):
        now = self.clock() if now is None else now
        while self.next_seq <= self.collector.schedule.count and now >= self.collector.schedule.decision(self.next_seq):
            seq = self.next_seq
            frame, window, reasons = self.collector.close(seq)
            decision = self.collector.schedule.decision(seq)
            meta = {"run_id": self.collector.run_id, "boot_id": self.collector.boot_id,
                    "sequence": seq, "decision_ns": decision,
                    "frame_build_lag_ms": (now-decision)/1e6,
                    "observed_router_count": int(frame["mask"].sum()),
                    "missing_router_count": int(10-frame["mask"].sum()),
                    "current_report_age_max_ms": float(frame["age"][frame["mask"]>0].max()*1000) if frame["mask"].any() else None,
                    "frame_sha256": hashlib.sha256(b"".join(frame[k].tobytes() for k in sorted(frame))).hexdigest()}
            self.audit({**meta, "event": "frame", "reasons": reasons,
                        "mask": frame["mask"].tolist(), "age": frame["age"].tolist(),
                        "age_valid": frame["age_valid"].tolist(),
                        "feature_valid": frame["feature_valid"].tolist()})
            self.inference.submit(window, meta)
            self.next_seq += 1
        self.inference.poll()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--start-ns", type=int, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-ms", type=float, default=500)
    args = parser.parse_args()
    if not 0 < args.timeout_ms < 3000:
        parser.error("Timeout must be positive and shorter than cadence")
    if hashlib.sha256((args.package / "registry.json").read_bytes()).hexdigest() != args.registry_sha256:
        raise ValueError("Pinned registry digest mismatch")
    import torch
    torch.set_num_threads(1)
    runtime, entry, registry_hash = load_runtime(args.package, args.model_id)
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    schedule = Schedule(args.start_ns, args.count)
    if time.monotonic_ns() >= args.start_ns:
        raise ValueError("Start must be future after runtime initialization")
    audit = JsonlAudit(args.output)
    provenance = {"registry_sha256": registry_hash,
                  "checkpoint_sha256": entry["checkpoint_sha256"],
                  "encoder_sha256": entry["encoder_sha256"], "threshold": .5}
    inference = AsyncInference(runtime, audit, args.model_id, timeout_ms=args.timeout_ms,
                               provenance=provenance)
    service = Service(CausalCollector(args.run_id, boot, schedule), inference, audit)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    bound = False
    try:
        # Never unlink an existing endpoint; fail if another service owns it.
        sock.bind(str(args.socket))
        bound = True
        args.socket.chmod(0o600)
        sock.setblocking(False)
        selector = selectors.DefaultSelector()
        selector.register(sock, selectors.EVENT_READ)
        audit({"event": "ready", "run_id": args.run_id, "boot_id": boot,
               "start_ns": args.start_ns, "model_id": args.model_id, **provenance})
        print(json.dumps({"event": "ready", "socket": str(args.socket), "boot_id": boot}), flush=True)
        finish = schedule.decision(args.count) + int(args.timeout_ms*1e6) + 200_000_000
        iterations, maximum_gap_ms, previous_loop = 0, 0.0, time.monotonic_ns()
        while time.monotonic_ns() < finish:
            now = time.monotonic_ns()
            maximum_gap_ms = max(maximum_gap_ms, (now-previous_loop)/1e6)
            previous_loop = now
            iterations += 1
            due = schedule.decision(service.next_seq) if service.next_seq <= args.count else finish
            timeout = min(.005, max(0, (due-now)/1e9))
            if selector.select(timeout):
                # Bounded batch: flood cannot starve decision timers.
                for _ in range(100):
                    try:
                        raw, _, flags, _ = sock.recvmsg(65536)
                    except BlockingIOError:
                        break
                    try:
                        if flags & socket.MSG_TRUNC:
                            raise ValueError("Truncated datagram")
                        service.ingest(json.loads(raw))
                    except (ValueError, KeyError, TypeError) as exc:
                        audit({"event": "report_rejected", "reason": str(exc),
                               "observed_ns": time.monotonic_ns()})
            service.tick()
            audit.check()
        service.tick()
        audit({"event": "complete", "frames": service.collector.closed_sequence,
               "event_loop_iterations": iterations, "max_event_loop_gap_ms": maximum_gap_ms})
    finally:
        try:
            inference.close()
        finally:
            sock.close()
            if bound:
                args.socket.unlink()
            audit.close()


if __name__ == "__main__":
    main()
