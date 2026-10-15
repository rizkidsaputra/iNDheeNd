#!/usr/bin/env python3
"""Bounded telemetry-only ML-7 smoke run on the existing Mini-NDN VM.

Requires root only for namespaces/topology. Creates its own processes/logs,
refuses an already active topology, and terminates only those it spawned.
"""
import argparse
from contextlib import suppress
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess as sp
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from ml7_online import ARRAYS, CausalCollector, ROUTERS, Schedule, load_runtime


def command(args,timeout=10):
    r=sp.run(list(map(str,args)),capture_output=True,text=True,timeout=timeout)
    if r.returncode: raise RuntimeError(f"Command failed {args}: {r.stderr}")
    return r.stdout


def rows(path):
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def counts():
    ps=command(["ps","-eo","args"])
    return [sum(bool(re.fullmatch(pattern,line)) for line in ps.splitlines())
            for pattern in (r"nfd --config /tmp/minindn/\w+/nfd.conf",r"nlsr -f /tmp/minindn/\w+/nlsr.conf")]


def pid(node):
    matches=command(["pgrep","-f",f"^nfd --config /tmp/minindn/{node}/nfd.conf$"]).split()
    if len(matches)!=1: raise RuntimeError("NFD namespace missing/ambiguous")
    return matches[0]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--python",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if os.geteuid()!=0: raise RuntimeError("Namespace smoke requires root")
    if counts()!=[0,0]: raise RuntimeError("Existing topology active; refusing automatic replacement")
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    old=Path("/home/ndn/research/experiments/poc11/scripts/poc11c")
    topo=Path("/home/ndn/research/experiments/iNDheeNd/minindn_topology.conf")
    kernel_before=command(["journalctl","-k","-b","--no-pager"])
    (out/"kernel_before.log").write_text(kernel_before)
    if "soft lockup" in kernel_before.lower(): raise RuntimeError("Kernel health preflight failed")
    streams=[]; processes=[]; reporters={}; topology=None; stopped=set()
    def launch(cmd,name):
        stream=open(out/name,"w"); streams.append(stream)
        process=sp.Popen(list(map(str,cmd)),stdout=stream,stderr=sp.STDOUT,start_new_session=True)
        processes.append(process); return process
    def ns(node,cmd):
        return ["mnexec","-a",pid(node),"env",f"NDN_CLIENT_TRANSPORT=unix:///run/nfd/{node}.sock",*cmd]
    health=[]; baseline_swap=0; busy_since=None
    old_cpu=list(map(int,Path("/proc/stat").read_text().splitlines()[0].split()[1:9]))
    last_health=time.monotonic_ns()
    def sample_health():
        nonlocal old_cpu,last_health,baseline_swap,busy_since
        now=time.monotonic_ns()
        mem={s.split(":")[0]:int(s.split()[1]) for s in Path("/proc/meminfo").read_text().splitlines() if len(s.split())>=2 and s.split()[1].isdigit()}
        current=list(map(int,Path("/proc/stat").read_text().splitlines()[0].split()[1:9]))
        total=sum(current)-sum(old_cpu); idle=current[3]+current[4]-old_cpu[3]-old_cpu[4]
        busy=100*(1-idle/total) if total>0 else 0
        swap=mem["SwapTotal"]-mem["SwapFree"]
        row={"monotonic_ns":now,"counts":counts(),"sample_gap_s":(now-last_health)/1e9,
             "mem_available_kb":mem["MemAvailable"],"swap_used_kb":swap,"cpu_busy_percent":busy}
        old_cpu=current; last_health=now; health.append(row)
        with (out/"health.jsonl").open("a") as f: f.write(json.dumps(row)+"\n")
        if len(health)==1: baseline_swap=swap
        if row["counts"]!=[21,21] or mem["MemAvailable"]<1536*1024 or swap>baseline_swap+1024:
            raise RuntimeError("Runtime health gate failed")
        if len(health)>1 and row["sample_gap_s"]>5: raise RuntimeError("Health sampling gap")
        if busy>85:
            if busy_since is None: busy_since=now
            elif now-busy_since>10_000_000_000: raise RuntimeError("Sustained CPU overload")
        else: busy_since=None
        if Path("/proc/sys/kernel/random/boot_id").read_text().strip()!=boot: raise RuntimeError("Boot changed")
    try:
        topology=launch(["env","PYTHONPATH=/home/ndn/research/simulators/mini-ndn", "/usr/bin/python3",old/"poc11c_topology_v1.py",topo,"--no-cli","--log-dir",out/"topology"],"topology_stdout.log")
        until=time.monotonic()+180
        while not (out/"topology/topology_ready.json").exists():
            if topology.poll() is not None or time.monotonic()>until: raise RuntimeError("Topology readiness failed")
            time.sleep(.2)
        until=time.monotonic()+40
        while True:
            fib=command(ns("m",["nfdc","fib","list"]))
            if all(f"/monitor/{r}/telemetry " in fib for r in ROUTERS): break
            if time.monotonic()>until: raise RuntimeError("Monitoring routes incomplete")
            time.sleep(1)
        (out/"m_fib.txt").write_text(fib)
        if counts()!=[21,21]: raise RuntimeError("Topology count mismatch")
        initial_pids={r:pid(r) for r in (*ROUTERS,"m")}
        initial_ticks={r:Path(f"/proc/{p}/stat").read_text().rpartition(") ")[2].split()[19] for r,p in initial_pids.items()}
        sample_health()
        start=time.monotonic_ns()+20_000_000_000; schedule=Schedule(start,8)
        run="ml7_telemetry_smoke"
        package=ROOT/"vendor/ml6_ndn_ifa_offline"
        registry_hash=hashlib.sha256((package/"registry.json").read_bytes()).hexdigest()
        endpoint=Path("/tmp/ml7-vm-m.sock")
        if endpoint.exists(): raise RuntimeError("Sidecar socket already exists")
        sidecar=launch([args.python,ROOT/"scripts/ml7_service.py","--package",package,
                        "--model-id","ml5a_mask_age_s42","--registry-sha256",registry_hash,
                        "--socket",endpoint,"--run-id",run,"--start-ns",start,"--count","8",
                        "--timeout-ms","500","--output",out/"decisions.jsonl"],"service_stdout.log")
        until=time.monotonic()+17
        while not any(e.get("event")=="ready" for e in rows(out/"decisions.jsonl")):
            if sidecar.poll() is not None or time.monotonic()>until: raise RuntimeError("Sidecar readiness failed")
            sample_health(); time.sleep(1)
        source=out/"source_payloads"; source.mkdir()
        binary=ROOT/"scripts/poc11c_apps_ml7_v8"
        for router in ROUTERS:
            (source/router).mkdir()
            instance=f"{boot}:{initial_pids[router]}:{initial_ticks[router]}"
            reporters[router]=launch(ns(router,["env",f"POC11C_NFD_INSTANCE={instance}",binary,"reporter",router,run,start,24,0,source]),f"reporter_{router}.jsonl")
        collector=launch(ns("m",["env",f"ML7_SOCKET={endpoint}",binary,"collector","m",run,start,24,0,",".join(ROUTERS)]),"collector_wire.jsonl")
        # Catch acquisition/readiness failures before consuming a single frame.
        while True:
            ready=all(any(e.get("event")=="prefix_registered" for e in rows(out/f"reporter_{r}.jsonl"))
                      and any(e.get("event")=="face_ready" for e in rows(out/f"reporter_{r}.jsonl")) for r in ROUTERS)
            ready=ready and any(e.get("event")=="face_ready" for e in rows(out/"collector_wire.jsonl"))
            if ready: break
            if any(p.poll() is not None for p in reporters.values()) or collector.poll() is not None:
                raise RuntimeError("Reporter/collector acquisition app exited before start")
            if time.monotonic_ns()>start-1_000_000_000: raise RuntimeError("Acquisition readiness margin failed")
            sample_health(); time.sleep(.5)
        config={"run_id":run,"boot_id":boot,"start_ns":start,"count":8,"duration_s":24,
                "model_id":"ml5a_mask_age_s42","registry_sha256":registry_hash,
                "binary_sha256":hashlib.sha256(binary.read_bytes()).hexdigest(),
                "topology_sha256":hashlib.sha256(topo.read_bytes()).hexdigest(),
                "traffic":"telemetry/NLSR only, no benign/attack/UDP workload",
                "planned_faults":["pause r2 reporter around seq6 until target+2.1s","terminate r3 reporter before seq7","terminate all reporters before seq8"]}
        (out/"config.json").write_text(json.dumps(config,indent=2))
        faults=[(schedule.target("r2",6)-50_000_000,"pause_r2"),
                (schedule.target("r2",6)+2_100_000_000,"resume_r2"),
                (schedule.target("r3",7)-50_000_000,"stop_r3"),
                (schedule.target("r1",8)-50_000_000,"blackout")]
        fault_pointer=0; next_health=time.monotonic_ns()+1_000_000_000
        until=schedule.decision(8)+1_000_000_000
        while time.monotonic_ns()<until:
            now=time.monotonic_ns()
            if now>=next_health:
                sample_health(); next_health=now+1_000_000_000
                for router,p in initial_pids.items():
                    if not Path(f"/proc/{p}").exists() or Path(f"/proc/{p}/stat").read_text().rpartition(") ")[2].split()[19]!=initial_ticks[router]:
                        raise RuntimeError("NFD instance changed")
            if sidecar.poll() is not None and now<schedule.decision(8): raise RuntimeError("Sidecar exited early")
            while fault_pointer<len(faults) and now>=faults[fault_pointer][0]:
                _,fault=faults[fault_pointer]
                if fault=="pause_r2": os.killpg(reporters["r2"].pid,signal.SIGSTOP)
                elif fault=="resume_r2": os.killpg(reporters["r2"].pid,signal.SIGCONT)
                elif fault=="stop_r3": reporters["r3"].terminate(); stopped.add("r3")
                else:
                    for router,p in reporters.items():
                        if p.poll() is None: p.terminate(); stopped.add(router)
                with (out/"faults.jsonl").open("a") as f: f.write(json.dumps({"event":fault,"monotonic_ns":now})+"\n")
                fault_pointer+=1
            time.sleep(.01)
        sidecar.wait(timeout=5)
        if sidecar.returncode: raise RuntimeError("Sidecar failed")
        collector.wait(timeout=8)
        if collector.returncode: raise RuntimeError("Collector failed")
        wire=[e for e in rows(out/"collector_wire.jsonl") if e.get("event")=="telemetry_received"]
        audit=rows(out/"decisions.jsonl"); frames=[e for e in audit if e["event"]=="frame"]
        decisions=[e for e in audit if e["event"]=="decision"]
        if len(frames)!=8 or len(decisions)!=8: raise RuntimeError("Decision count mismatch")
        byseq={d["sequence"]:d for d in decisions}
        if frames[5]["mask"][1]!=0 or frames[6]["mask"][2]!=0 or any(frames[7]["mask"]): raise RuntimeError("Planned fault masks mismatch")
        if byseq[8]["reason"]!="no_current_reports": raise RuntimeError("Blackout fallback mismatch")
        if any(byseq[i]["status"]!="prediction" for i in (5,6,7)): raise RuntimeError("Usable windows not inferred")
        if any(e.get("event")=="ml7_ipc_drop" for e in rows(out/"collector_wire.jsonl")): raise RuntimeError("Unexpected IPC loss")
        import torch,numpy as np,torch_geometric
        torch.set_num_threads(1)
        runtime,_,_=load_runtime(package,"ml5a_mask_age_s42")
        ledger=CausalCollector(run,boot,schedule); ordered=sorted(wire,key=lambda e:e["monotonic_ns"]); pointer=0; errors=[]; windows=[]
        for seq in range(1,9):
            while pointer<len(ordered) and ordered[pointer]["monotonic_ns"]<=schedule.decision(seq):
                e=ordered[pointer]; ledger.receive(e["raw_payload"],e["monotonic_ns"]); pointer+=1
            f,w,_=ledger.close(seq)
            digest=hashlib.sha256(b"".join(f[k].tobytes() for k in sorted(f))).hexdigest()
            if digest!=frames[seq-1]["frame_sha256"]: raise RuntimeError("Online/collector replay tensor mismatch")
            if w is not None: windows.append(w)
            if seq in (5,6,7):
                p,y=runtime.predict({k:v.copy() for k,v in w.items()}); d=byseq[seq]
                errors.append(abs(float(p[0])-d["probability_ifa"]))
                if errors[-1]>2e-6 or int(y[0])!=d["prediction"]: raise RuntimeError("Online/replay probability mismatch")
        np.savez_compressed(out/"live_windows.npz",**{k:np.concatenate([w[k] for w in windows]) for k in ARRAYS})
        kernel_after=command(["journalctl","-k","-b","--no-pager"]); (out/"kernel_after.log").write_text(kernel_after)
        if "soft lockup" in kernel_after.lower(): raise RuntimeError("Kernel health failure")
        prediction_rows=[d for d in decisions if d["status"]=="prediction"]
        summary={"status":"LIVE_SMOKE_PASSED","frames":8,"decisions":8,"predictions":3,
                 "received_reports":len(wire),"max_probability_error":max(errors),
                 "inference_p95_ms":float(np.percentile([d["inference_ms"] for d in prediction_rows],95)),
                 "decision_to_publish_p95_ms":float(np.percentile([d["decision_to_publish_ms"] for d in prediction_rows],95)),
                 "sample_end_to_publish_max_ms":max(d["sample_end_to_publish_max_ms"] for d in prediction_rows),
                 "frame_build_lag_max_ms":max(f["frame_build_lag_ms"] for f in frames),
                 "minimum_mem_available_kb":min(h["mem_available_kb"] for h in health),
                 "maximum_swap_used_kb":max(h["swap_used_kb"] for h in health),
                 "maximum_health_gap_s":max(h["sample_gap_s"] for h in health[1:]),
                 "runtime":{"python":sys.version,"torch":torch.__version__,"numpy":np.__version__,"torch_geometric":torch_geometric.__version__},
                 "limitations":["Telemetry-only smoke, no detection labels", "Reporter pause/termination are controlled faults, not congestion", "Only three predictions; latency smoke only", "Independent ML-8 robustness not established"]}
        (out/"live_summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary),flush=True)
    except Exception as exc:
        (out/"failure.json").write_text(json.dumps({"status":"FAILED","error":str(exc)},indent=2))
        raise
    finally:
        for p in reversed(processes):
            if p.poll() is None:
                with suppress(ProcessLookupError): os.killpg(p.pid,signal.SIGCONT)
                p.terminate()
                try: p.wait(timeout=8)
                except sp.TimeoutExpired: os.killpg(p.pid,signal.SIGKILL); p.wait(timeout=3)
        for stream in streams: stream.close()
        # A topology process may receive SIGTERM before its signal handler is
        # installed. Initial preflight proved no NFD/NLSR existed; remove only
        # processes with this topology's exact temporary configuration paths.
        if topology is not None and counts()!=[0,0]:
            listing=command(["ps","-eo","pid=,args="])
            for line in listing.splitlines():
                parts=line.strip().split(maxsplit=1)
                if len(parts)==2 and re.fullmatch(r"(?:nfd --config /tmp/minindn/\w+/nfd.conf|nlsr -f /tmp/minindn/\w+/nlsr.conf)",parts[1]):
                    with suppress(ProcessLookupError): os.kill(int(parts[0]),signal.SIGTERM)
            time.sleep(.5)
        (out/"cleanup.json").write_text(json.dumps({"remaining_counts":counts(),"spawned_processes_exited":all(p.poll() is not None for p in processes)},indent=2))


if __name__=="__main__": main()
