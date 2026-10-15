#!/usr/bin/env python3

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
RAW = ROOT / "data/extracted/dataset_development"
ML1 = ROOT / "data/processed/ml1"
ML2 = ROOT / "data/processed/ml2"
OUT = ROOT / "results/ml2"

OUT.mkdir(parents=True, exist_ok=True)

schema = json.loads(
    (ROOT / "results/ml1/feature_schema.json").read_text()
)
graph = json.loads(
    (ROOT / "configs/topology/router_graph.json").read_text()
)

routers = schema["router_order"]
features = schema["derived_model_feature_candidates"]

counters = schema["raw_features"][:8]
snapshots = schema["raw_features"][8:]

checks = []
run_results = []


def check(name, condition):
    if not bool(condition):
        raise AssertionError(f"FAILED: {name}")
    checks.append(name)
    print(f"[PASS] {name}")


def jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def same(a, b, name):
    np.testing.assert_allclose(
        a, b, rtol=1e-5, atol=1e-5,
        equal_nan=True, err_msg=name
    )
    checks.append(name)


def validate_graph():
    print("\n=== GRAPH VALIDATION ===")

    source = ROOT / "configs/topology/minindn_topology.conf"

    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    check(
        "Topology SHA-256",
        digest == graph["topology_sha256"]
    )

    index = {r: i for i, r in enumerate(routers)}
    observed = set()
    in_links = False

    for line in source.read_text().splitlines():
        line = line.strip()

        if line == "[links]":
            in_links = True
            continue

        if line.startswith("[") and line.endswith("]"):
            in_links = False
            continue

        if not in_links or not line or line.startswith("#"):
            continue

        names = line.split()[0].lower().split(":")

        if len(names) != 2:
            raise AssertionError("Invalid topology link")

        a, b = names

        if a in index and b in index:
            observed.add((index[a], index[b]))
            observed.add((index[b], index[a]))

    edge = np.asarray(graph["edge_index"], dtype=np.int64)

    check("Graph shape", edge.shape == (2, 20))
    check("Router count", len(routers) == 10)
    check("Router order", routers == graph["router_order"])

    check(
        "Edges match original topology",
        set(map(tuple, edge.T.tolist())) == observed
    )

    return edge


def expected_label(gt, decision):
    intervals = gt.get("raw", {}).get(
        "attack_emission_intervals", []
    )

    return int(any(
        int(x["start_ns"]) <= decision <= int(
            x.get("stop_scheduled_ns", x["last_submit_ns"])
        )
        for x in intervals
    ))


def validate_run(path, edge):
    run_id = path.stem
    print(f"\n=== RUN: {run_id} ===")

    csv_path = ML1 / f"{run_id}.csv"
    telemetry_path = (
        RAW / "telemetry_collector" / f"{run_id}.jsonl"
    )
    availability_path = (
        RAW / "telemetry_collector"
        / f"{run_id}.availability.jsonl"
    )
    gt_path = RAW / "ground_truth" / f"{run_id}.json"

    for p in [
        csv_path, telemetry_path,
        availability_path, gt_path
    ]:
        check(f"{run_id}: {p.name} exists", p.is_file())

    gt = json.loads(gt_path.read_text())
    df = pd.read_csv(csv_path)

    df["router_id"] = pd.Categorical(
        df["router_id"],
        categories=routers,
        ordered=True
    )

    df = df.sort_values(
        ["sequence", "router_id"]
    ).reset_index(drop=True)

    check(f"{run_id}: CSV rows", len(df) == 500)
    check(
        f"{run_id}: unique router-time pairs",
        not df.duplicated(["sequence", "router_id"]).any()
    )

    observed_keys = set(
        zip(df["sequence"], df["router_id"].astype(str))
    )
    expected_keys = {
        (seq, r)
        for seq in range(1, 51)
        for r in routers
    }

    check(
        f"{run_id}: complete time grid",
        observed_keys == expected_keys
    )

    reports = {
        (r["sequence"], r["router_id"]): r
        for r in jsonl(telemetry_path)
    }

    avail = {
        (r["sequence"], r["router_id"]): r
        for r in jsonl(availability_path)
    }

    check(
        f"{run_id}: availability grid",
        set(avail) == expected_keys
    )

    check(
        f"{run_id}: ground truth identity",
        gt["run_id"] == run_id
    )

    # Reconstruct features chronologically.
    # Never use late, unavailable or future telemetry.
    previous = {}
    reconstructed = []
    masks = []
    ages = []
    labels = []

    for _, row in df.iterrows():
        seq = int(row["sequence"])
        router = str(row["router_id"])
        key = (seq, router)

        a = avail[key]
        report = reports.get(key)

        mask = int(
            report is not None
            and report["on_time"] is True
        )

        assert int(a["mask"]) == mask
        assert int(row["mask"]) == mask

        raw_age = a.get("age_seconds")
        age = (
            float(raw_age)
            if raw_age is not None
            else np.nan
        )

        csv_age = row["age_seconds"]

        if np.isnan(age):
            assert pd.isna(csv_age)
        else:
            assert np.isclose(age, csv_age, atol=1e-5)

        decision = int(a["decision_ns"])
        assert int(row["decision_ns"]) == decision

        label = expected_label(gt, decision)
        assert int(row["label_attack_active"]) == label

        values = {f: np.nan for f in features}

        if mask:
            assert report["received_ns"] <= decision
            assert report["created_ns"] <= report["received_ns"]

            for name in snapshots:
                values[name] = float(report["metrics"][name])

            old = previous.get(router)

            if old is not None:
                old_report, old_seq = old

                elapsed = (
                    report["created_ns"]
                    - old_report["created_ns"]
                ) / 1e9

                same_instance = (
                    report.get("nfd_instance_id")
                    == old_report.get("nfd_instance_id")
                )

                deltas = {
                    c: (
                        report["metrics"][c]
                        - old_report["metrics"][c]
                    )
                    for c in counters
                }

                reset = (
                    not same_instance
                    or elapsed <= 0
                    or any(v < 0 for v in deltas.values())
                )

                if not reset:
                    for c in counters:
                        values[f"{c}_rate"] = (
                            deltas[c] / elapsed
                        )

            previous[router] = (report, seq)

        # Verify ML-1 did not use noncausal values.
        for name in features:
            expected = values[name]
            actual = row[name]

            if np.isnan(expected):
                assert pd.isna(actual), (
                    f"{run_id}, {key}: "
                    f"unexpected {name}={actual}"
                )
            else:
                assert np.isclose(
                    actual, expected,
                    rtol=1e-5, atol=1e-5
                ), (
                    f"{run_id}, {key}: "
                    f"{name} mismatch"
                )

        reconstructed.append([
            values[name] for name in features
        ])
        masks.append(mask)
        ages.append(age)
        labels.append(label)

    expected_x = np.asarray(
        reconstructed, dtype=np.float32
    ).reshape(50, 10, 10)

    expected_m = np.asarray(
        masks, dtype=np.int64
    ).reshape(50, 10)

    expected_a = np.asarray(
        ages, dtype=np.float32
    ).reshape(50, 10)

    expected_y_nodes = np.asarray(
        labels, dtype=np.int64
    ).reshape(50, 10)

    check(
        f"{run_id}: graph label consistency",
        np.all(expected_y_nodes == expected_y_nodes[:, :1])
    )

    expected_y = expected_y_nodes[:, 0]

    with np.load(path, allow_pickle=False) as data:
        xraw = data["X_raw"]
        x = data["X"]
        mask = data["mask"]
        age = data["age"]
        age_valid = data["age_valid"]
        valid = data["feature_valid"]
        y = data["y"]
        edges = data["edge_index"]

        check(f"{run_id}: X shape", x.shape == (50, 10, 10))
        check(f"{run_id}: Mask shape", mask.shape == (50, 10))
        check(f"{run_id}: Age shape", age.shape == (50, 10))
        check(f"{run_id}: label shape", y.shape == (50,))
        check(f"{run_id}: edges", np.array_equal(edges, edge))

        same(xraw, expected_x, f"{run_id}: causal features")

        check(
            f"{run_id}: mask exact",
            np.array_equal(mask, expected_m)
        )

        same(
            age,
            np.nan_to_num(expected_a, nan=0.0),
            f"{run_id}: age alignment"
        )

        check(
            f"{run_id}: age validity",
            np.array_equal(
                age_valid,
                np.isfinite(expected_a).astype(np.float32)
            )
        )

        check(
            f"{run_id}: feature validity",
            np.array_equal(
                valid,
                np.isfinite(expected_x).astype(np.float32)
            )
        )

        check(
            f"{run_id}: label exact",
            np.array_equal(y, expected_y)
        )

        check(
            f"{run_id}: zero placeholders",
            np.array_equal(
                x,
                np.where(np.isfinite(xraw), xraw, 0.0)
            )
        )

        check(
            f"{run_id}: no unavailable features",
            np.isnan(xraw[mask == 0]).all()
        )

        check(
            f"{run_id}: finite model tensor",
            np.isfinite(x).all()
        )

        # Verify every 5-step window against its source.
        windows = 50 - 5 + 1

        for i in range(windows):
            sl = slice(i, i + 5)

            for source, target in [
                ("X", "X_window"),
                ("mask", "mask_window"),
                ("age", "age_window"),
                ("age_valid", "age_valid_window"),
                ("feature_valid", "feature_valid_window"),
            ]:
                assert np.array_equal(
                    data[target][i],
                    data[source][sl]
                ), f"{run_id}: window mismatch {target}"

            assert data["y_window"][i] == y[i + 4]

        check(
            f"{run_id}: temporal windows causal",
            data["X_window"].shape == (46, 5, 10, 10)
        )

        summary = {
            "run_id": run_id,
            "timesteps": 50,
            "windows": windows,
            "mask_one": int(mask.sum()),
            "mask_zero": int((mask == 0).sum()),
            "attack_timesteps": int(y.sum()),
            "attack_windows": int(data["y_window"].sum()),
        }

    print("SUMMARY:", summary)
    return summary


def main():
    edge = validate_graph()

    files = sorted(ML2.glob("*.npz"))
    check("Six NPZ runs", len(files) == 6)

    for path in files:
        run_results.append(validate_run(path, edge))

    check(
        "Total windows",
        sum(r["windows"] for r in run_results) == 276
    )

    check(
        "Total on-time",
        sum(r["mask_one"] for r in run_results) == 2859
    )

    check(
        "Total unavailable",
        sum(r["mask_zero"] for r in run_results) == 141
    )

    result = {
        "status": "PASSED",
        "checks_passed": len(checks),
        "total_runs": len(run_results),
        "total_windows": 276,
        "runs": run_results,
        "limitations": [
            "No training-run-only scaler fitted yet.",
            "No cross-run holdout protocol established yet.",
            "Mixed protocol versions require controlled evaluation.",
            "IFA label validates configured attack interval, not packet-by-packet delivery.",
            "Counter rates across missing intervals summarize multiple periods.",
            "Mask/Age representation is not yet experimentally validated.",
        ],
    }

    (OUT / "ml2_validation.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8"
    )

    print("\n============================")
    print("ML-2 VALIDATION: PASSED")
    print("Checks passed:", len(checks))
    print("Runs:", len(run_results))
    print("Windows:", result["total_windows"])
    print("============================")


if __name__ == "__main__":
    main()
