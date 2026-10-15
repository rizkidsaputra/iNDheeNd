#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / "results/ml5d"

config = json.loads(
    (ROOT / "configs/ml5d/reciprocal.json").read_text()
)

validation = json.loads(
    (OUT / "ml5d_validation.json").read_text()
)

assert validation["status"] == "PASSED"

m = pd.read_csv(
    OUT / "metrics_by_run.csv"
)

p = pd.read_csv(
    OUT / "all_predictions.csv"
)

# Only original, unmodified holdout predictions
# are used for the principal comparison.
original = m[
    (m["split"] == "holdout")
    & (m["condition"] == "original")
    & (m["run_id"] == "ALL")
].copy()

assert len(original) == 18

aggregate = original.groupby(
    ["direction", "variant"],
    as_index=False
).agg(
    f1_mean=("f1", "mean"),
    f1_std=("f1", "std"),
    recall_mean=("recall", "mean"),
    precision_mean=("precision", "mean"),
    fpr_mean=("fpr", "mean"),
    ap_mean=("ap", "mean"),
    roc_auc_mean=("roc_auc", "mean"),
)

aggregate.to_csv(
    OUT / "reciprocal_aggregate.csv",
    index=False
)

print("=== 1. RECIPROCAL HOLDOUT COMPARISON ===")

print(
    aggregate.round(4).to_string(index=False)
)

print("\n=== 2. INDIVIDUAL SEED RESULTS ===")

print(
    original[
        [
            "direction", "seed", "variant",
            "f1", "recall", "fpr", "ap",
            "tn", "fp", "fn", "tp",
        ]
    ].sort_values(
        ["direction", "seed", "variant"]
    ).round(4).to_string(index=False)
)

print("\n=== 3. B1 FALSE POSITIVES ===")

b1 = m[
    (m["split"] == "holdout")
    & (m["condition"] == "original")
    & (m["run_id"].str.contains("_b1"))
]

print(
    b1[
        [
            "direction", "seed",
            "variant", "n", "fp", "fpr",
        ]
    ].sort_values(
        ["direction", "seed", "variant"]
    ).to_string(index=False)
)

print("\n=== 4. ATTACK BOUNDARY ===")

boundary = p[
    (p["split"] == "holdout")
    & (p["condition"] == "original")
    & (p["run_id"].str.contains("_a1"))
    & (p["end_sequence"].isin(
        [16, 17, 20, 21, 30, 31]
    ))
].copy()

boundary.to_csv(
    OUT / "boundary_predictions.csv",
    index=False
)

table = boundary.pivot_table(
    index=[
        "direction",
        "end_sequence",
        "label",
    ],
    columns="variant",
    values="probability",
    aggfunc="mean",
)

print(table.round(4).to_string())

print("\n=== 5. AGE INTERVENTION DIAGNOSTIC ===")

intervention = m[
    (m["split"] == "holdout")
    & (m["run_id"] == "ALL")
].copy()

summary = intervention.pivot_table(
    index=[
        "direction", "seed", "variant"
    ],
    columns="condition",
    values=["f1", "recall", "fpr"],
    aggfunc="first",
)

print(summary.round(4).to_string())

print("\n=== 6. PAIRED AGE PROBABILITY SHIFT ===")

original_p = p[
    (p["split"] == "holdout")
    & (p["condition"] == "original")
].copy()

modified_p = p[
    (p["split"] == "holdout")
    & (p["condition"] == "nominal_missing")
].copy()

keys = [
    "direction", "seed", "variant",
    "run_id", "end_sequence", "label",
]

paired = original_p.merge(
    modified_p,
    on=keys,
    suffixes=("_original", "_modified"),
    validate="one_to_one",
)

assert len(paired) == 18 * 92

paired["delta_probability"] = (
    paired["probability_modified"]
    - paired["probability_original"]
)

paired["abs_delta"] = (
    paired["delta_probability"].abs()
)

paired["prediction_flip"] = (
    paired["prediction_original"]
    != paired["prediction_modified"]
)

shift = paired.groupby(
    ["direction", "variant"],
    as_index=False
).agg(
    windows=("abs_delta", "size"),
    mean_abs_probability_shift=("abs_delta", "mean"),
    max_abs_probability_shift=("abs_delta", "max"),
    prediction_flips=("prediction_flip", "sum"),
)

shift.to_csv(
    OUT / "age_intervention_shift.csv",
    index=False
)

paired.to_csv(
    OUT / "age_intervention_paired.csv",
    index=False
)

print(shift.round(5).to_string(index=False))

# Neutralizing Age must not change Plain/Mask.
for variant in ["plain", "mask"]:
    subset = paired[
        paired["variant"] == variant
    ]

    np.testing.assert_allclose(
        subset["probability_original"],
        subset["probability_modified"],
        rtol=1e-6,
        atol=1e-6,
    )

print("[PASS] Plain and Mask are unaffected by Age intervention")

# The M0 holdout has Mask=1 throughout:
# no Age replacements should occur.
m0_holdout = paired[
    paired["direction"] == "m1_to_m0"
]

np.testing.assert_allclose(
    m0_holdout["probability_original"],
    m0_holdout["probability_modified"],
    rtol=1e-6,
    atol=1e-6,
)

print("[PASS] M0 holdout unchanged by missing-slot intervention")

print("\n=== 7. LATE ATTACK-PHASE RECOVERY ===")

late_phase = paired[
    paired["run_id"].str.contains("_a1")
    & paired["end_sequence"].between(21, 30)
].copy()

late_phase_results = late_phase.groupby(
    ["direction", "variant"],
    as_index=False
).agg(
    n=("label", "size"),
    detected_original=("prediction_original", "sum"),
    detected_modified=("prediction_modified", "sum"),
    mean_probability_original=(
        "probability_original", "mean"
    ),
    mean_probability_modified=(
        "probability_modified", "mean"
    ),
)

late_phase_results.to_csv(
    OUT / "late_phase_recovery.csv",
    index=False
)

print(
    late_phase_results.round(4).to_string(
        index=False
    )
)

print("\nML-5D DIAGNOSTICS COMPLETED")
