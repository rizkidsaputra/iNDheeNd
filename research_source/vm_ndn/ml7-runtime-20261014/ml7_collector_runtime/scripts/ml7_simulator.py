#!/usr/bin/env python3
"""Real-time integration on a Linux clock with C++ IPC and frozen ML-6 model.

Synthetic NFD-shaped reports only. Does not start an NDN workload or evaluate
detection robustness. Uses 3 s cadence and real deadline timers, not sleep-free
offline replay. Run in a fresh output directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ml7_online import CausalCollector, FEATURES, ROUTERS, Schedule, load_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--model-id", default="ml5a_mask_age_s42")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    directory = args.output_dir.resolve()
    # Short AF_UNIX pathname independent of a potentially long output path.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="ml7_") as temporary:
        temp = Path(temporary)
        endpoint = temp / "m.sock"
        source = temp / "sender.cpp"
        source.write_text('#include "ml7_datagram_bridge.hpp"\nint main(int n,char**v){ Ml7DatagramBridge b(v[1]); return b.send(v[2]) ? 0 : 9; }\n')
        subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-I", str(ROOT/"src"),
                        str(source), "-o", str(temp/"sender")], check=True, capture_output=True)
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        schedule = Schedule(time.monotonic_ns()+20_000_000_000, 7)
        registry_hash = hashlib.sha256((args.package / "registry.json").read_bytes()).hexdigest()
        command = [sys.executable, str(ROOT/"scripts/ml7_service.py"),
                   "--package", str(args.package.resolve()), "--model-id", args.model_id,
                   "--registry-sha256", registry_hash, "--socket", str(endpoint),
                   "--run-id", "simulator", "--start-ns", str(schedule.start_ns),
                   "--count", "7", "--output", str(directory/"decisions.jsonl")]
        stderr_file = open(directory/"service_stderr.log", "w")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr_file, text=True)
        records, sends = [], []
        try:
            # Bounded readiness polling; never wait indefinitely on stdout.
            until = time.monotonic()+17
            while not endpoint.exists():
                if process.poll() is not None or time.monotonic() > until:
                    raise RuntimeError("Service did not become ready")
                time.sleep(.01)
            def send(event):
                begin = time.perf_counter_ns()
                subprocess.run([str(temp/"sender"), str(endpoint), json.dumps(event)], check=True)
                sends.append((time.perf_counter_ns()-begin)/1e6)
            events=[]
            for seq in range(1,7):
                for router in ROUTERS:
                    if seq == 4 and router == "r3": continue
                    lag = 2_100_000_000 if seq == 6 and router == "r2" else 300_000_000
                    events.append((schedule.target(router,seq)+lag,seq,router))
            for due,seq,router in sorted(events):
                while time.monotonic_ns()<due:
                    time.sleep(min(.01,(due-time.monotonic_ns())/1e9))
                created = schedule.target(router,seq)+10_000_000
                received = time.monotonic_ns()
                p={"run_id":"simulator","boot_id":boot,"router_id":router,"sequence":seq,
                   "created_ns":created,"nfd_instance_id":boot+":simulated:1",
                   "metrics":{k:seq*100+i for i,k in enumerate(FEATURES)}}
                event={"event":"telemetry_received","router_id":router,"sequence":seq,
                       "monotonic_ns":received,"raw_payload":json.dumps(p),
                       "data_name":f"/monitor/{router}/telemetry/poc11c/simulator/{seq}"}
                send(event); records.append({"payload":p,"received_ns":received})
                if seq == 1 and router == "r1":
                    send(event)  # duplicate must be rejected, not overwrite
                    bad=dict(event); bad["data_name"]+="/bad"; send(bad)
                    bad=dict(event); q=dict(p); q["boot_id"]="wrong-boot"; bad["raw_payload"]=json.dumps(q); send(bad)
                    send([1,2,3])  # valid JSON with invalid outer schema
            process.wait(timeout=10)
            if process.returncode:
                raise RuntimeError(f"Service exited {process.returncode}")
        finally:
            if process.poll() is None:
                process.terminate(); process.wait(timeout=5)
            stderr_file.close()
        (directory/"reports.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
        audit=[json.loads(s) for s in (directory/"decisions.jsonl").read_text().splitlines()]
        frames=[e for e in audit if e["event"]=="frame"]
        decisions=[e for e in audit if e["event"]=="decision"]
        if len(frames)!=7 or len(decisions)!=7 or len({d["sequence"] for d in decisions})!=7:
            raise ValueError("Missing/duplicate frame or decision")
        by_seq={d["sequence"]:d for d in decisions}
        if [by_seq[i]["reason"] for i in range(1,5)] != ["window_warmup"]*4:
            raise ValueError("Warmup mismatch")
        if by_seq[7]["reason"]!="no_current_reports" or by_seq[7]["prediction"] is not None:
            raise ValueError("Blackout did not abstain")
        if frames[3]["mask"][2]!=0 or frames[5]["mask"][1]!=0 or frames[5]["feature_valid"][1]!=[0]*10:
            raise ValueError("Missing/late report leaked features")
        rejects=[e for e in audit if e["event"]=="report_rejected"]
        if len(rejects)!=4:
            raise ValueError("Invalid reports not rejected")
        if sum(e.get("disposition")=="late_report_excluded" for e in audit)!=1:
            raise ValueError("Late report not audited")
        import torch
        torch.set_num_threads(1)
        runtime,_,_=load_runtime(args.package,args.model_id)
        collector=CausalCollector("simulator",boot,schedule)
        pointer=0; errors=[]
        records.sort(key=lambda r:r["received_ns"])
        for seq in range(1,8):
            while pointer<len(records) and records[pointer]["received_ns"]<=schedule.decision(seq):
                r=records[pointer]; collector.receive(r["payload"],r["received_ns"]); pointer+=1
            f,w,_=collector.close(seq)
            digest=hashlib.sha256(b"".join(f[k].tobytes() for k in sorted(f))).hexdigest()
            if digest!=frames[seq-1]["frame_sha256"]: raise ValueError("Online frame differs from causal reference")
            if seq in (5,6):
                p,y=runtime.predict({k:v.copy() for k,v in w.items()}); d=by_seq[seq]
                if d["status"]!="prediction" or d["prediction"]!=int(y[0]):
                    raise ValueError("Inference decision mismatch")
                error=abs(float(p[0])-d["probability_ifa"]); errors.append(error)
                if error>2e-6: raise ValueError("Online probability mismatch")
        predictions=[d for d in decisions if d["status"]=="prediction"]
        result={"status":"SIMULATOR_INTEGRATION_PASSED","frames":7,"decisions":7,
                "predictions":len(predictions),"invalid_reports_rejected":len(rejects),
                "max_probability_error":max(errors),
                "inference_p95_ms":float(np.percentile([d["inference_ms"] for d in predictions],95)),
                "decision_to_publish_p95_ms":float(np.percentile([d["decision_to_publish_ms"] for d in predictions],95)),
                "sample_end_to_publish_max_ms":max(d["sample_end_to_publish_max_ms"] for d in predictions),
                "frame_build_lag_max_ms":max(f["frame_build_lag_ms"] for f in frames),
                "cpp_sender_max_ms_including_process_launch":max(sends),
                "event_loop":next(e for e in audit if e["event"]=="complete"),
                "model_id":args.model_id,"registry_sha256":registry_hash,
                "limitations":["Synthetic metrics, no NFD/NDN traffic", "Only two model decisions; latency is a smoke measurement", "Full ndn-cxx collector compilation and VM run remain separate gates"]}
        (directory/"integration_summary.json").write_text(json.dumps(result,indent=2))
        print(json.dumps(result))


if __name__=="__main__": main()
