#!/usr/bin/env python3
"""
ML-1: Dataset Audit & Causal Feature Engineering
PoC 11C — NDN IFA Robust Detection

Input  : data/extracted/dataset_development/
Output : data/processed/ml1/, results/ml1/

Raw evidence is read-only. No ML training.
"""

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path.cwd()
BASE = ROOT / "data/extracted/dataset_development"
OUT = ROOT / "data/processed/ml1"
REPORT = ROOT / "results/ml1"

COUNTERS = [
    "nInInterests", "nOutInterests",
    "nInData", "nOutData",
    "nInNacks", "nOutNacks",
    "nSatisfiedInterests",
    "nUnsatisfiedInterests",
]
SNAPSHOTS = ["nPitEntries", "nCsEntries"]
FEATURES = COUNTERS + SNAPSHOTS

RATE_NAMES = [f"{x}_rate" for x in COUNTERS]
FEATURE_NAMES = RATE_NAMES + SNAPSHOTS

EXPECTED = {
    "safety150_b0_m0_v8": (500, 500, 0, 0),
    "safety150_b1_m0_v8": (500, 500, 0, 0),
    "safety150_a1_m0_v8": (500, 500, 0, 0),
    "m1dev_v14_r01_b0": (500, 461, 29, 10),
    "m1dev_v16_r01_b1": (500, 452, 36, 12),
    "m1dev_v16_r02_a1": (500, 446, 42, 12),
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def warn(warnings, msg):
    warnings.append(msg)
    print("  [WARN]", msg)


def save_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)

    topology = read_json(BASE / "topology/metadata.json")
    routers = topology["router_ids"]

    assert len(routers) == 10
    assert len(set(routers)) == 10
    assert topology["feature_schema"] == FEATURES

    manifests = {
        x["run_id"]: x
        for x in read_jsonl(BASE / "manifest/run_index.jsonl")
    }

    collector = BASE / "telemetry_collector"
    detected = {
        p.name.removesuffix(".jsonl")
        for p in collector.glob("*.jsonl")
        if not p.name.endswith(".availability.jsonl")
    }

    assert detected == set(EXPECTED), (
        f"Unexpected or missing runs: {detected ^ set(EXPECTED)}"
    )

    summary = []
    warnings = []
    all_rows = []

    print("\n=== ML-1 DATASET AUDIT ===")

    for run_id in EXPECTED:
        print(f"\nRUN: {run_id}")

        tele_path = collector / f"{run_id}.jsonl"
        avail_path = collector / f"{run_id}.availability.jsonl"
        gt_path = BASE / "ground_truth" / f"{run_id}.json"

        telemetry = read_jsonl(tele_path)
        availability = read_jsonl(avail_path)
        gt = read_json(gt_path)
        manifest = manifests.get(run_id)

        assert manifest is not None
        assert len(availability) == 500
        assert gt["run_id"] == run_id

        mode = gt["mode"]
        impairment = gt["M"]

        assert manifest["mode"] == mode
        assert manifest["M"] == impairment

        expected_keys = {
            (seq, router)
            for seq in range(1, 51)
            for router in routers
        }

        avail_by_key = {}
        for a in availability:
            key = (a["sequence"], a["router_id"])
            assert key not in avail_by_key, (
                f"Duplicate availability: {run_id} {key}"
            )
            avail_by_key[key] = a

        assert set(avail_by_key) == expected_keys

        tele_by_key = {}
        for t in telemetry:
            key = (t["sequence"], t["router_id"])
            assert key not in tele_by_key, (
                f"Duplicate telemetry: {run_id} {key}"
            )
            assert set(t["metrics"]) == set(FEATURES)
            tele_by_key[key] = t

        assert set(tele_by_key) <= expected_keys

        on_time = sum(bool(t["on_time"]) for t in telemetry)
        received = len(telemetry)
        late = received - on_time
        not_received = 500 - received
        mask_zero = sum(a["mask"] == 0 for a in availability)

        expected_total, expected_on, expected_late, expected_nr = (
            EXPECTED[run_id]
        )

        assert (
            len(availability),
            on_time,
            late,
            not_received,
        ) == (
            expected_total,
            expected_on,
            expected_late,
            expected_nr,
        )

        assert mask_zero == late + not_received

        for key, a in avail_by_key.items():
            t = tele_by_key.get(key)
            should_mask = int(
                t is not None and t["on_time"] is True
            )
            assert a["mask"] == should_mask, (
                f"Mask mismatch: {run_id} {key}"
            )

        for field, actual in [
            ("generated", 500),
            ("received", received),
            ("on_time", on_time),
            ("late", late),
            ("not_received", not_received),
        ]:
            if manifest.get(field) != actual:
                warn(
                    warnings,
                    f"{run_id}: manifest {field} mismatch"
                )

        # Ground truth is for labels, never model features.
        intervals = gt.get("raw", {}).get(
            "attack_emission_intervals", []
        )

        if mode == "A1" and not intervals:
            raise ValueError(
                f"A1 missing attack intervals: {run_id}"
            )

        # Earlier raw confirmation fields can be placeholders.
        # Do not equate placeholder false with an independently
        # disproven attack. This requires separate QA review.
        confirmation = gt.get("raw", {}).get(
            "confirmation_status", "unknown"
        )

        if mode == "A1":
            warn(
                warnings,
                f"{run_id}: independently verify A1 confirmation "
                f"using QA/evidence; raw status={confirmation}"
            )

        # Only previous eligible on-time collector observations.
        previous = {}
        run_rows = []

        for seq in range(1, 51):
            for router in routers:
                key = (seq, router)
                a = avail_by_key[key]
                t = tele_by_key.get(key)

                decision_ns = a["decision_ns"]
                mask = int(a["mask"])
                age = a.get("age_seconds")

                if age is not None:
                    assert math.isfinite(float(age))
                    assert float(age) >= 0

                if mask == 1:
                    assert t is not None
                    assert t["received_ns"] <= decision_ns, (
                        f"Noncausal arrival: {run_id} {key}"
                    )
                    assert t["created_ns"] <= t["received_ns"]

                # Timestamp alignment, not scenario label,
                # determines attack-active classification.
                attack_active = any(
                    int(interval["start_ns"]) <= decision_ns
                    <= int(
                        interval.get(
                            "stop_scheduled_ns",
                            interval["last_submit_ns"],
                        )
                    )
                    for interval in intervals
                )

                row = {
                    "run_id": run_id,
                    "protocol_stratum": manifest.get(
                        "protocol_stratum", ""
                    ),
                    "mode_audit": mode,
                    "impairment_audit": impairment,
                    "router_id": router,
                    "sequence": seq,
                    "decision_ns": decision_ns,
                    "label_attack_active": int(attack_active),
                    "mask": mask,
                    "age_seconds": age,
                    "last_on_time_sequence": a.get(
                        "last_on_time_sequence"
                    ),
                    "created_ns_observed": "",
                    "received_ns_observed": "",
                    "rate_elapsed_s": "",
                    "counter_gap": "",
                    "counter_reset": 0,
                }

                for feature in FEATURE_NAMES:
                    row[feature] = ""

                if mask:
                    current_ns = t["created_ns"]
                    row["created_ns_observed"] = current_ns
                    row["received_ns_observed"] = t["received_ns"]

                    # Instantaneous snapshots are valid only when
                    # the relevant report is observed on time.
                    for feature in SNAPSHOTS:
                        row[feature] = t["metrics"][feature]

                    prev = previous.get(router)

                    if prev is not None:
                        prev_t, prev_seq = prev

                        same_instance = (
                            t.get("nfd_instance_id")
                            == prev_t.get("nfd_instance_id")
                        )

                        elapsed = (
                            current_ns - prev_t["created_ns"]
                        ) / 1e9

                        deltas = {
                            name: (
                                t["metrics"][name]
                                - prev_t["metrics"][name]
                            )
                            for name in COUNTERS
                        }

                        reset = (
                            not same_instance
                            or elapsed <= 0
                            or any(d < 0 for d in deltas.values())
                        )

                        row["counter_gap"] = seq - prev_seq
                        row["counter_reset"] = int(reset)

                        if not reset:
                            row["rate_elapsed_s"] = elapsed

                            # Rates span the observed interval.
                            # They are not instantaneous estimates.
                            for name in COUNTERS:
                                row[f"{name}_rate"] = (
                                    deltas[name] / elapsed
                                )

                    previous[router] = (t, seq)

                run_rows.append(row)

        assert len(run_rows) == 500

        columns = list(run_rows[0].keys())
        save_csv(OUT / f"{run_id}.csv", run_rows, columns)
        all_rows.extend(run_rows)

        attack_slots = sum(
            r["label_attack_active"] for r in run_rows
        )

        run_summary = {
            "run_id": run_id,
            "protocol_stratum": manifest.get(
                "protocol_stratum"
            ),
            "mode": mode,
            "M": impairment,
            "scheduled": 500,
            "received": received,
            "on_time": on_time,
            "late": late,
            "not_received": not_received,
            "mask_zero": mask_zero,
            "attack_labeled_router_slots": attack_slots,
            "counter_reset_slots": sum(
                int(r["counter_reset"]) for r in run_rows
            ),
            "sha256_telemetry": sha256(tele_path),
            "sha256_availability": sha256(avail_path),
        }

        summary.append(run_summary)

        print(
            f"  received={received} "
            f"on_time={on_time} late={late} "
            f"not_received={not_received}"
        )
        print(
            f"  mask_zero={mask_zero} "
            f"attack_router_slots={attack_slots}"
        )

    # Audit fields stay available for tracing, but
    # must never automatically enter the model tensor.
    save_csv(
        OUT / "all_runs_audit_features.csv",
        all_rows,
        list(all_rows[0].keys())
    )

    schema = {
        "router_order": routers,
        "raw_features": FEATURES,
        "derived_model_feature_candidates": FEATURE_NAMES,
        "additional_inputs": ["mask", "age_seconds"],
        "rate_policy": (
            "Nonnegative delta between consecutive on-time "
            "collector observations of the same NFD instance, "
            "divided by actual source sampling elapsed time. "
            "First observation and resets remain missing."
        ),
        "missing_policy": (
            "No source audit, future value or unobserved "
            "telemetry is used for imputation."
        ),
        "label_policy": (
            "Temporal attack-active label using workload "
            "ground-truth intervals evaluated at decision_ns."
        ),
        "excluded_from_model_inputs": [
            "run_id",
            "mode_audit",
            "impairment_audit",
            "protocol_stratum",
            "label_attack_active",
            "router_id",
            "sequence",
            "decision_ns",
            "created_ns_observed",
            "received_ns_observed",
            "last_on_time_sequence",
        ],
        "model_readiness": (
            "Feature engineering candidate only; graph topology, "
            "normalization, label QA and causal tests remain gates."
        )
    }

    (REPORT / "feature_schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )

    result = {
        "run_count": len(summary),
        "scheduled_total": sum(x["scheduled"] for x in summary),
        "on_time_total": sum(x["on_time"] for x in summary),
        "late_total": sum(x["late"] for x in summary),
        "not_received_total": sum(
            x["not_received"] for x in summary
        ),
        "runs": summary,
        "warnings": warnings,
    }

    (REPORT / "audit_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    with (REPORT / "audit_report.md").open(
        "w", encoding="utf-8"
    ) as f:
        f.write("# ML-1 Dataset Audit\n\n")
        f.write(
            "Development-only dataset; not a final "
            "generalization evaluation.\n\n"
        )
        f.write(
            "| Run | On-time | Late | Not received "
            "| Mask=0 | Attack-labeled router slots |\n"
        )
        f.write(
            "|---|---:|---:|---:|---:|---:|\n"
        )
        for s in summary:
            f.write(
                f"| {s['run_id']} | {s['on_time']} "
                f"| {s['late']} | {s['not_received']} "
                f"| {s['mask_zero']} "
                f"| {s['attack_labeled_router_slots']} |\n"
            )
        f.write(
            "\n## Remaining scientific gates\n\n"
            "- Verify actual topology edges against Mini-NDN.\n"
            "- Verify independent attack confirmation in QA.\n"
            "- Audit differences between v8, v14 and v16.\n"
            "- Fit normalization using training runs only.\n"
            "- Define causal missing-value handling.\n"
            "- Do not use run metadata or labels as features.\n"
            "- Do not claim generalization from one run per cell.\n"
        )

    print("\n=== ML-1 SUMMARY ===")
    print(json.dumps({
        "runs": result["run_count"],
        "scheduled": result["scheduled_total"],
        "on_time": result["on_time_total"],
        "late": result["late_total"],
        "not_received": result["not_received_total"],
        "warnings": len(warnings)
    }, indent=2))

    print("\nOutput:")
    print(" ", OUT)
    print(" ", REPORT)


if __name__ == "__main__":
    main()
