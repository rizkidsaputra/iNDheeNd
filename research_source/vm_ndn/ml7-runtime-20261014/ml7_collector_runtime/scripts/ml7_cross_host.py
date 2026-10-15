#!/usr/bin/env python3
"""Compare live VM decisions with the original WSL frozen ML-6 runtime."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from ml7_online import load_runtime


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--package",type=Path,required=True)
    p.add_argument("--evidence",type=Path,required=True)
    args=p.parse_args()
    import torch
    torch.set_num_threads(1)
    runtime,_,registry_hash=load_runtime(args.package,"ml5a_mask_age_s42")
    decisions=[json.loads(s) for s in (args.evidence/"decisions.jsonl").read_text().splitlines()]
    decisions=[d for d in decisions if d.get("event")=="decision" and d["status"]=="prediction"]
    with np.load(args.evidence/"live_windows.npz",allow_pickle=False) as d:
        arrays={k:d[k].copy() for k in d.files}
    rows=[]
    for decision in decisions:
        i=decision["sequence"]-5
        probability,label=runtime.predict({k:v[i:i+1] for k,v in arrays.items()})
        error=abs(float(probability[0])-decision["probability_ifa"])
        if error>2e-6 or int(label[0])!=decision["prediction"]:
            raise ValueError("CPU runtime cross-host parity failure")
        rows.append({"sequence":decision["sequence"],"vm_probability_ifa":decision["probability_ifa"],
                     "wsl_probability_ifa":float(probability[0]),"max_abs_error":error})
    if len(rows)!=3: raise ValueError("Expected exactly three live predictions")
    result={"status":"PASSED","compared":len(rows),"max_abs_error":max(r["max_abs_error"] for r in rows),
            "registry_sha256":registry_hash,"rows":rows,"limitation":"Numerical/runtime compatibility only"}
    (args.evidence/"cross_host_parity.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result))


if __name__=="__main__": main()
