import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
ML1 = ROOT / "data/processed/ml1"
ML2 = ROOT / "data/processed/ml2"
REPORT = ROOT / "results/ml2"

ML2.mkdir(parents=True, exist_ok=True)
REPORT.mkdir(parents=True, exist_ok=True)

schema = json.loads(
    (ROOT / "results/ml1/feature_schema.json").read_text()
)

graph = json.loads(
    (ROOT / "configs/topology/router_graph.json").read_text()
)

routers = schema["router_order"]
features = schema["derived_model_feature_candidates"]
edge_index = np.asarray(graph["edge_index"], dtype=np.int64)

assert len(routers) == 10
assert len(features) == 10
assert graph["router_order"] == routers
assert edge_index.shape == (2, 20)

# Verify graph provenance.
topology_path = ROOT / "configs/topology/minindn_topology.conf"
topology_hash = hashlib.sha256(topology_path.read_bytes()).hexdigest()

assert topology_hash == graph["topology_sha256"], (
    "Topology source hash mismatch"
)

# Verify bidirectional edge representation.
edges = set(map(tuple, edge_index.T.tolist()))
assert len(edges) == 20
assert all((v, u) in edges for u, v in edges)
assert all(u != v for u, v in edges)

WINDOW = 5
summaries = []

files = sorted(
    p for p in ML1.glob("*.csv")
    if p.name != "all_runs_audit_features.csv"
)

assert len(files) == 6

for path in files:
    df = pd.read_csv(path)
    run_id = path.stem

    assert len(df) == 500
    assert df["run_id"].nunique() == 1
    assert df["run_id"].iloc[0] == run_id

    assert not df.duplicated(
        ["sequence", "router_id"]
    ).any()

    expected_pairs = {
        (s, r)
        for s in range(1, 51)
        for r in routers
    }

    actual_pairs = set(
        zip(df["sequence"], df["router_id"])
    )
    assert actual_pairs == expected_pairs

    df["router_id"] = pd.Categorical(
        df["router_id"],
        categories=routers,
        ordered=True
    )

    df = df.sort_values(
        ["sequence", "router_id"]
    ).reset_index(drop=True)

    # Exact alignment: timestep x router.
    X_raw = df[features].to_numpy(
        dtype=np.float32
    ).reshape(50, 10, 10)

    M = df["mask"].to_numpy(
        dtype=np.int64
    ).reshape(50, 10)

    A = df["age_seconds"].to_numpy(
        dtype=np.float32
    ).reshape(50, 10)

    decision_ns = df["decision_ns"].to_numpy(
        dtype=np.int64
    ).reshape(50, 10)

    labels_by_router = df["label_attack_active"].to_numpy(
        dtype=np.int64
    ).reshape(50, 10)

    # Binary graph-level label must agree across routers.
    assert np.all(
        labels_by_router == labels_by_router[:, :1]
    )

    y = labels_by_router[:, 0]

    assert set(np.unique(M)).issubset({0, 1})
    assert set(np.unique(y)).issubset({0, 1})

    # No unavailable report is allowed to supply features.
    assert np.isnan(X_raw[M == 0]).all(), (
        f"Missing report contains features: {run_id}"
    )

    # Snapshot metrics must exist for valid reports.
    assert np.isfinite(
        X_raw[M == 1, -2:]
    ).all(), f"Missing PIT/CS snapshot: {run_id}"

    # Check that every observed report arrived by decision.
    for _, row in df[df["mask"] == 1].iterrows():
        assert (
            row["created_ns_observed"]
            <= row["received_ns_observed"]
            <= row["decision_ns"]
        ), f"Noncausal report: {run_id}"

    # Keep missing feature flags separate.
    V = np.isfinite(X_raw).astype(np.float32)

    # Zeros are placeholders, not observations.
    # V and M distinguish them from real zero measurements.
    X = np.where(
        np.isfinite(X_raw), X_raw, 0.0
    ).astype(np.float32)

    # Age originates from collector availability metadata.
    # Do not interpret a missing age as zero age.
    A_valid = np.isfinite(A).astype(np.float32)
    A_filled = np.where(
        np.isfinite(A), A, 0.0
    ).astype(np.float32)

    assert (A[np.isfinite(A)] >= 0).all()

    # Build temporal windows within each run only.
    Xw, Mw, Aw = [], [], []
    Vw, AVw, yw = [], [], []

    for end in range(WINDOW - 1, 50):
        start = end - WINDOW + 1

        Xw.append(X[start:end + 1])
        Mw.append(M[start:end + 1])
        Aw.append(A_filled[start:end + 1])
        Vw.append(V[start:end + 1])
        AVw.append(A_valid[start:end + 1])

        # Prediction target = current decision timestep.
        yw.append(y[end])

    Xw = np.stack(Xw)
    Mw = np.stack(Mw)
    Aw = np.stack(Aw)
    Vw = np.stack(Vw)
    AVw = np.stack(AVw)
    yw = np.asarray(yw, dtype=np.int64)

    assert X.shape == (50, 10, 10)
    assert M.shape == (50, 10)
    assert A_filled.shape == (50, 10)
    assert V.shape == (50, 10, 10)

    assert Xw.shape == (46, 5, 10, 10)
    assert Mw.shape == (46, 5, 10)
    assert Aw.shape == (46, 5, 10)
    assert Vw.shape == (46, 5, 10, 10)
    assert yw.shape == (46,)

    np.savez_compressed(
        ML2 / f"{run_id}.npz",
        X=X,
        X_raw=X_raw,
        mask=M,
        age=A_filled,
        age_valid=A_valid,
        feature_valid=V,
        y=y,
        decision_ns=decision_ns,
        X_window=Xw,
        mask_window=Mw,
        age_window=Aw,
        age_valid_window=AVw,
        feature_valid_window=Vw,
        y_window=yw,
        edge_index=edge_index,
    )

    summaries.append({
        "run_id": run_id,
        "timesteps": 50,
        "nodes": 10,
        "features": 10,
        "windows": 46,
        "on_time": int(M.sum()),
        "missing": int((M == 0).sum()),
        "positive_timesteps": int(y.sum()),
        "positive_windows": int(yw.sum()),
        "undefined_feature_values": int(
            (V == 0).sum()
        ),
        "age_missing_values": int(
            (A_valid == 0).sum()
        ),
    })

    print(
        f"[OK] {run_id}: "
        f"X={X.shape}, "
        f"windows={Xw.shape}, "
        f"attack_windows={int(yw.sum())}"
    )

metadata = {
    "window_size": WINDOW,
    "router_order": routers,
    "feature_order": features,
    "edge_index": edge_index.tolist(),
    "topology_sha256": topology_hash,
    "notes": [
        "No train/validation/test split performed.",
        "No scaler has been fitted.",
        "No future temporal windows are used.",
        "Missing feature values are zero-filled only in X.",
        "feature_valid and mask retain missingness.",
        "Age is separate; missing ages have age_valid=0.",
        "Labels and audit metadata are not model inputs.",
        "Per-run windows must not be randomly split.",
    ],
    "runs": summaries,
}

(REPORT / "ml2_summary.json").write_text(
    json.dumps(metadata, indent=2)
)

print("\n=== ML-2 SUMMARY ===")
print("Runs:", len(summaries))
print("Total windows:", sum(s["windows"] for s in summaries))
print("Topology edges:", edge_index.shape[1])
print("Saved:", ML2)
print("Report:", REPORT / "ml2_summary.json")
