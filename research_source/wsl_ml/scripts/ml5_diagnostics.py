#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
    average_precision_score,
)

ROOT = Path.cwd()
DATA = ROOT / "data/processed/ml2"
OUT = ROOT / "results/ml5"
CONFIG = json.loads(
    (ROOT / "configs/ml5/ablation.json").read_text()
)

p = pd.read_csv(
    OUT / "all_predictions.csv"
)

expected_models = (
    len(CONFIG["variants"]) * len(CONFIG["seeds"])
)
assert expected_models == 9

assert len(p) == expected_models * 276

threshold = CONFIG["threshold"]

print("=== 1. BOUNDARY ERRORS ===")

boundary = p[
    (p["split"] == "holdout")
    & (p["run_id"] == "m1dev_v16_r02_a1")
    & (p["end_sequence"].isin([16, 17, 30, 31]))
].copy()

boundary = boundary.sort_values(
    ["seed", "variant", "end_sequence"]
)

boundary.to_csv(
    OUT / "boundary_predictions.csv",
    index=False
)

print(
    boundary[
        [
            "seed", "variant", "end_sequence",
            "label", "probability", "prediction"
        ]
    ].to_string(index=False)
)

print("\n=== 2. HOLDOUT ERROR PATTERNS ===")

h = p[p["split"] == "holdout"].copy()
h["error"] = h["label"] != h["prediction"]

errors = h[h["error"]].copy()

errors.to_csv(
    OUT / "holdout_errors.csv",
    index=False
)

print("Errors by variant and seed:")

print(
    h.groupby(["variant", "seed"])["error"]
    .sum().to_string()
)

print("\nErrors by missingness at decision:")

h["missing_group"] = np.where(
    h["missing_at_decision"] > 0,
    "missing_present",
    "all_reports_on_time"
)

diagnostics = h.groupby(
    ["variant", "missing_group"]
).agg(
    windows=("error", "size"),
    errors=("error", "sum"),
    positives=("label", "sum"),
).reset_index()

diagnostics.to_csv(
    OUT / "missingness_diagnostics.csv",
    index=False
)

print(diagnostics.to_string(index=False))

print("\n=== 3. OBSERVATION-ONLY SHORTCUT PROBE ===")


def collect(run_ids):
    feature_sets = {
        "mask_only": [],
        "age_only": [],
        "mask_age": [],
        "sequence_only": [],
    }

    labels = []

    for run_id in run_ids:
        with np.load(
            DATA / f"{run_id}.npz",
            allow_pickle=False,
        ) as d:
            mask = d["mask_window"]
            age = d["age_window"]
            valid = d["age_valid_window"]
            y = d["y_window"]

            # Fixed 5 x 10 temporal observations.
            mask_flat = mask.reshape(len(y), -1)
            age_flat = np.log1p(
                age
            ).reshape(len(y), -1)

            valid_flat = valid.reshape(
                len(y), -1
            )

            # End sequence 5 ... 50.
            sequence = np.arange(
                5, 51
            ).reshape(-1, 1)

            feature_sets["mask_only"].append(
                mask_flat
            )

            feature_sets["age_only"].append(
                np.concatenate(
                    [age_flat, valid_flat],
                    axis=1
                )
            )

            feature_sets["mask_age"].append(
                np.concatenate(
                    [mask_flat, age_flat, valid_flat],
                    axis=1
                )
            )

            feature_sets["sequence_only"].append(
                sequence
            )

            labels.append(y)

    x = {
        name: np.concatenate(chunks)
        for name, chunks in feature_sets.items()
    }

    y = np.concatenate(labels)
    return x, y


train_x, train_y = collect(
    CONFIG["train_runs"]
)

holdout_x, holdout_y = collect(
    CONFIG["holdout_runs"]
)

assert len(train_y) == 184
assert len(holdout_y) == 92
assert train_y.sum() == 15
assert holdout_y.sum() == 15

probe_results = []

for name in train_x:
    # StandardScaler and classifier only fit on training.
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )
    )

    probe.fit(train_x[name], train_y)

    probabilities = probe.predict_proba(
        holdout_x[name]
    )[:, 1]

    predicted = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        holdout_y, predicted,
        labels=[0, 1]
    ).ravel()

    result = {
        "probe": name,
        "n": int(len(holdout_y)),
        "f1": float(f1_score(
            holdout_y, predicted,
            zero_division=0
        )),
        "precision": float(precision_score(
            holdout_y, predicted,
            zero_division=0
        )),
        "recall": float(recall_score(
            holdout_y, predicted,
            zero_division=0
        )),
        "average_precision": float(
            average_precision_score(
                holdout_y, probabilities
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    probe_results.append(result)

    print(
        f"{name}: "
        f"F1={result['f1']:.4f}, "
        f"Recall={result['recall']:.4f}, "
        f"AP={result['average_precision']:.4f}"
    )

(OUT / "shortcut_probes.json").write_text(
    json.dumps(probe_results, indent=2)
)

print("\n=== 4. ABLATION SUMMARY ===")

summary = pd.read_csv(
    OUT / "holdout_aggregate.csv"
)

print(summary.round(4).to_string(index=False))

print("\n=== 5. DIAGNOSTIC COMPLETED ===")
print("Boundary predictions saved")
print("Holdout errors saved")
print("Missingness diagnostics saved")
print("Shortcut probes saved")
