#!/usr/bin/env python3
"""Record final unit evidence and frozen artifact audit without modifying ML-6."""
import hashlib
import argparse
import json
from pathlib import Path
import platform
import sys
import unittest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--audit-only",action="store_true")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=root/"results/ml7"
    output.mkdir(parents=True,exist_ok=True)
    if args.audit_only:
        summary=json.loads((output/"unit_summary.json").read_text())
    else:
        suite=unittest.defaultTestLoader.discover(str(root/"tests"),pattern="test_ml7_online.py")
        with (output/"unit_tests.log").open("w") as stream:
            result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        summary={"status":"PASSED" if result.wasSuccessful() and not result.skipped else "FAILED_OR_SKIPPED",
                 "tests_run":result.testsRun,"failures":len(result.failures),
                 "errors":len(result.errors),"skipped":len(result.skipped)}
        (output/"unit_summary.json").write_text(json.dumps(summary,indent=2))
    frozen=json.loads((output/"freeze_before.json").read_text())
    violations=[p for p,digest in frozen.items() if hashlib.sha256((root/p).read_bytes()).hexdigest()!=digest]
    if violations: raise ValueError(f"Freeze violation: {violations}")
    import torch,torch_geometric,numpy
    environment={"python":platform.python_version(),"torch":torch.__version__,
                 "torch_geometric":torch_geometric.__version__,"numpy":numpy.__version__,
                 "platform":platform.platform(),"device":"cpu","torch_threads_service":1}
    (output/"environment.json").write_text(json.dumps(environment,indent=2))
    inventory={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
               for directory,pattern in (("src","ml7*"),("scripts","ml7*.py"),("scripts","poc11c_apps_ml7_v8.cpp"),
                                          ("tests","test_ml7_online.py"),("docs","ml7_contract.md"))
               for p in (root/directory).glob(pattern) if p.is_file()}
    (output/"implementation_sha256.json").write_text(json.dumps(inventory,indent=2))
    acceptance="PENDING_LIVE_AUDIT"
    live=output/"vm_live/live_summary.json"
    if live.exists():
        l=json.loads(live.read_text())
        cleanup=json.loads((output/"vm_live/cleanup.json").read_text())
        replay=json.loads((output/"replay_audit.json").read_text())
        parity=json.loads((output/"vm_live/cross_host_parity.json").read_text())
        if l["status"]=="LIVE_SMOKE_PASSED" and cleanup["remaining_counts"]==[0,0] and cleanup["spawned_processes_exited"] and replay["status"]=="REPLAY_PASSED" and parity["status"]=="PASSED" and summary["status"]=="PASSED":
            acceptance="ML7_ENGINEERING_PASSED"
    audit={"unit_tests":summary,"frozen_files_verified":len(frozen),"freeze_violations":violations,
           "full_ml7_acceptance":acceptance,
           "robustness":"NOT_ESTABLISHED"}
    (output/"local_audit.json").write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit))
    if summary["status"]!="PASSED": raise SystemExit(1)


if __name__=="__main__": main()
