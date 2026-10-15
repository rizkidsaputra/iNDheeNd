#!/usr/bin/env python3
"""Independent causal replay from collector received rows; NPZ only is oracle.

This uses development evidence, not independent robustness evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ml7_online import ARRAYS, CausalCollector, Schedule, frozen_inventory, load_runtime


def replay_run(repo, run_id):
    directory = repo / "data/extracted/dataset_development/telemetry_collector"
    reports = [json.loads(s) for s in (directory / f"{run_id}.jsonl").read_text().splitlines() if s]
    availability = [json.loads(s) for s in (directory / f"{run_id}.availability.jsonl").read_text().splitlines() if s]
    # Only the prespecified decision timeline is taken from availability, never
    # its Mask/Age/last_sequence fields. No source or ground truth files read.
    decisions = sorted({a["decision_ns"] for a in availability})
    if len(decisions) != 50 or np.diff(decisions).tolist() != [3_000_000_000]*49:
        raise ValueError("Unexpected replay cadence")
    schedule = Schedule(decisions[0] - 2_800_000_000, len(decisions))
    boot = reports[0]["nfd_instance_id"].split(":")[0]
    collector = CausalCollector(run_id, boot, schedule)
    ordered = sorted(reports, key=lambda p: p["received_ns"])
    pointer, frames, windows, dispositions = 0, [], [], {}
    for seq, decision in enumerate(decisions, 1):
        while pointer < len(ordered) and ordered[pointer]["received_ns"] <= decision:
            p = dict(ordered[pointer])
            p["boot_id"] = p["nfd_instance_id"].split(":")[0]
            disposition = collector.receive(p, p["received_ns"])
            dispositions[disposition] = dispositions.get(disposition, 0) + 1
            pointer += 1
        frame, window, _ = collector.close(seq)
        frames.append(frame)
        if window is not None:
            windows.append(window)
    actual = {k: np.concatenate([w[k] for w in windows]) for k in ARRAYS}
    mapping = {"x": "X_window", **{k: f"{k}_window" for k in ARRAYS if k != "x"}}
    errors = {}
    with np.load(repo / f"data/processed/ml2/{run_id}.npz", allow_pickle=False) as reference:
        for key, source in mapping.items():
            np.testing.assert_array_equal(actual[key], reference[source], err_msg=f"{run_id}:{key}")
            errors[key] = float(np.max(np.abs(actual[key] - reference[source])))
    return actual, {"run_id": run_id, "frames": len(frames), "windows": len(windows),
                    "tensor_max_abs_errors": errors, "report_dispositions": dispositions}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    repo = args.repo.resolve()
    package = repo / "dist/ml6_ndn_ifa_offline"
    before = frozen_inventory(package)
    runs = [p.stem for p in sorted((repo / "data/processed/ml2").glob("*.npz"))]
    arrays, rows = {}, []
    for run_id in runs:
        arrays[run_id], row = replay_run(repo, run_id)
        rows.append(row)
    registry = json.loads((package / "registry.json").read_text())
    prediction_rows = []
    for entry in registry["models"]:
        if entry["variant"] != "mask_age":
            continue
        runtime, _, _ = load_runtime(package, entry["id"])
        for run_id in runs:
            # Reference uses frozen ML-6 reader; actual uses raw report replay.
            ref = runtime.read_input(repo / f"data/processed/ml2/{run_id}.npz")
            ref_p, ref_y = runtime.predict(ref)
            # One window at a time exercises operational batch shape.
            p, y = [], []
            for i in range(46):
                pi, yi = runtime.predict({k: v[i:i+1] for k, v in arrays[run_id].items()})
                p.append(float(pi[0])); y.append(int(yi[0]))
            error = float(np.max(np.abs(np.asarray(p) - ref_p)))
            if error > 2e-6 or not np.array_equal(y, ref_y):
                raise ValueError(f"Replay prediction mismatch: {entry['id']} {run_id} {error}")
            prediction_rows.append({"model_id": entry["id"], "run_id": run_id,
                                    "compared": len(p), "max_abs_error": error})
    if frozen_inventory(package) != before:
        raise ValueError("ML-6 package changed")
    frozen = json.loads((repo / "results/ml7/freeze_before.json").read_text())
    mismatches = [p for p, digest in frozen.items()
                  if hashlib.sha256((repo / p).read_bytes()).hexdigest() != digest]
    if mismatches:
        raise ValueError(f"Freeze violations: {mismatches}")
    report = {"status": "REPLAY_PASSED", "runs": rows, "prediction_checks": prediction_rows,
              "probabilities_compared": sum(r["compared"] for r in prediction_rows),
              "max_probability_error": max(r["max_abs_error"] for r in prediction_rows),
              "freeze_files_verified": len(frozen), "freeze_violations": mismatches,
              "limitation": "Previously examined development runs; no robustness or live-network claim"}
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("runs", "prediction_checks")}))


if __name__ == "__main__":
    main()
