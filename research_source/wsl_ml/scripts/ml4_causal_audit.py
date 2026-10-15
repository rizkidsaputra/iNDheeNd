#!/usr/bin/env python3

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path.cwd()

RAW = ROOT / "data/extracted/dataset_development"
ML2 = ROOT / "data/processed/ml2"
OUT = ROOT / "results/ml4"

OUT.mkdir(parents=True, exist_ok=True)

ROUTERS = [f"r{i}" for i in range(1, 11)]


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def main():
    results = []
    checks = 0

    files = sorted(ML2.glob("*.npz"))
    assert len(files) == 6

    for npz_path in files:
        run_id = npz_path.stem

        telemetry_path = (
            RAW / "telemetry_collector"
            / f"{run_id}.jsonl"
        )

        availability_path = (
            RAW / "telemetry_collector"
            / f"{run_id}.availability.jsonl"
        )

        reports_list = read_jsonl(telemetry_path)
        availability_list = read_jsonl(
            availability_path
        )

        reports = {
            (r["sequence"], r["router_id"]): r
            for r in reports_list
        }

        availability = {
            (r["sequence"], r["router_id"]): r
            for r in availability_list
        }

        assert len(reports) == len(reports_list)
        assert len(availability) == len(availability_list)
        assert len(availability) == 500

        with np.load(
            npz_path,
            allow_pickle=False
        ) as data:
            masks = data["mask"].copy()
            ages = data["age"].copy()
            age_valid = data["age_valid"].copy()
            features = data["X_raw"].copy()
            validity = data["feature_valid"].copy()
            decisions = data["decision_ns"].copy()

        last_ontime = {}
        missing_with_stale_age = 0
        no_previous_observation = 0
        late_excluded = 0
        run_checks = 0

        for seq in range(1, 51):
            for ri, router in enumerate(ROUTERS):
                key = (seq, router)

                a = availability[key]
                r = reports.get(key)

                mask = int(a["mask"])

                expected_mask = int(
                    r is not None
                    and r["on_time"] is True
                )

                assert mask == expected_mask
                assert int(masks[seq-1, ri]) == mask

                decision_ns = int(a["decision_ns"])

                assert (
                    int(decisions[seq-1, ri])
                    == decision_ns
                )

                # Only current on-time reports update
                # last available observation.
                if mask == 1:
                    assert r is not None
                    assert int(r["created_ns"]) <= int(
                        r["received_ns"]
                    )
                    assert int(
                        r["received_ns"]
                    ) <= decision_ns

                    last_ontime[router] = r

                elif r is not None:
                    # Late telemetry must not update state.
                    assert r["on_time"] is False
                    late_excluded += 1

                previous = last_ontime.get(router)

                declared_last = a.get(
                    "last_on_time_sequence"
                )

                if previous is None:
                    assert declared_last in (None, 0)

                    assert a.get("age_seconds") is None
                    assert age_valid[seq-1, ri] == 0
                    assert ages[seq-1, ri] == 0

                    no_previous_observation += 1

                else:
                    assert int(declared_last) == int(
                        previous["sequence"]
                    )

                    assert (
                        int(previous["sequence"]) <= seq
                    )

                    expected_age = (
                        decision_ns -
                        int(previous["created_ns"])
                    ) / 1e9

                    assert expected_age >= 0

                    actual_age = a.get("age_seconds")
                    assert actual_age is not None

                    np.testing.assert_allclose(
                        float(actual_age),
                        expected_age,
                        rtol=1e-6,
                        atol=1e-5,
                    )

                    np.testing.assert_allclose(
                        float(ages[seq-1, ri]),
                        expected_age,
                        rtol=1e-6,
                        atol=1e-5,
                    )

                    assert age_valid[seq-1, ri] == 1

                    if mask == 0:
                        missing_with_stale_age += 1

                # Model features cannot be reconstructed
                # from reports unavailable at deadline.
                if mask == 0:
                    assert np.isnan(
                        features[seq-1, ri]
                    ).all()

                    assert (
                        validity[seq-1, ri] == 0
                    ).all()

                run_checks += 1

        assert run_checks == 500
        checks += run_checks

        result = {
            "run_id": run_id,
            "slot_checks": run_checks,
            "late_reports_excluded": late_excluded,
            "missing_with_valid_stale_age":
                missing_with_stale_age,
            "slots_without_previous_observation":
                no_previous_observation,
            "status": "PASSED",
        }

        results.append(result)

        print(f"\n[PASS] {run_id}")
        print("  Slots checked:", run_checks)
        print("  Late excluded:", late_excluded)
        print(
            "  Missing with valid Age:",
            missing_with_stale_age
        )
        print(
            "  No previous observation:",
            no_previous_observation
        )

    report = {
        "status": "PASSED",
        "runs": len(results),
        "slot_checks": checks,
        "late_reports_excluded": sum(
            r["late_reports_excluded"]
            for r in results
        ),
        "missing_with_valid_stale_age": sum(
            r["missing_with_valid_stale_age"]
            for r in results
        ),
        "results": results,
    }

    (OUT / "causal_audit.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n============================")
    print("ML-4 CAUSAL AUDIT: PASSED")
    print("Runs:", len(results))
    print("Router-time checks:", checks)
    print("============================")


if __name__ == "__main__":
    main()
