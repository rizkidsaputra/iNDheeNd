#!/usr/bin/env python3

import json
from pathlib import Path

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

aggregate = pd.read_csv(
    OUT / "reciprocal_aggregate.csv"
)

metrics = pd.read_csv(
    OUT / "metrics_by_run.csv"
)

recovery = pd.read_csv(
    OUT / "late_phase_recovery.csv"
)

shift = pd.read_csv(
    OUT / "age_intervention_shift.csv"
)

audit = json.loads(
    (OUT / "split_and_scaler_audit.json").read_text()
)

lines = [
    "# ML-5D — Reciprocal Controlled Training",
    "",
    "## 1. Status",
    "",
    "**Implementation and validation: PASSED**",
    "",
    "- Directions: 2",
    "- Models trained: 18",
    "- Variants: Plain, Mask, Mask+Age",
    "- Seeds: 42, 52, 62",
    "- Epochs: 20",
    "- Parameters per model: 35,009",
    "- Windows: 92 train and 92 holdout per direction",
    "- Checkpoint selection: final epoch",
    "- Threshold: fixed at 0.5",
    "",
    "## 2. Dataset Splits and Normalization",
    "",
]

for item in audit:
    lines.extend([
        f"### {item['direction']}",
        "",
        f"- Training: {', '.join(item['train_runs'])}",
        f"- Holdout: {', '.join(item['holdout_runs'])}",
        f"- Training positives: {item['training_positives']}",
        f"- Holdout positives: {item['holdout_positives']}",
        f"- Positive weight: {item['pos_weight']:.6f}",
        f"- Training on-time Age median: "
        f"{item['training_nominal_age_s']:.6f} s",
        "- Feature and Age scalers fitted on training runs only.",
        "",
    ])

lines.extend([
    "## 3. Original Holdout Comparison",
    "",
    "| Direction | Variant | F1 mean | Recall mean | FPR mean | AP mean |",
    "|---|---|---:|---:|---:|---:|",
])

for _, row in aggregate.iterrows():
    lines.append(
        f"| {row['direction']} "
        f"| {row['variant']} "
        f"| {row['f1_mean']:.4f} "
        f"| {row['recall_mean']:.4f} "
        f"| {row['fpr_mean']:.4f} "
        f"| {row['ap_mean']:.4f} |"
    )

lines.extend([
    "",
    "## 4. Per-Seed Results",
    "",
    "| Direction | Seed | Variant | F1 | Recall | FPR | AP |",
    "|---|---:|---|---:|---:|---:|---:|",
])

individual = metrics[
    (metrics["split"] == "holdout")
    & (metrics["condition"] == "original")
    & (metrics["run_id"] == "ALL")
].sort_values(
    ["direction", "seed", "variant"]
)

for _, row in individual.iterrows():
    lines.append(
        f"| {row['direction']} "
        f"| {int(row['seed'])} "
        f"| {row['variant']} "
        f"| {row['f1']:.4f} "
        f"| {row['recall']:.4f} "
        f"| {row['fpr']:.4f} "
        f"| {row['ap']:.4f} |"
    )

lines.extend([
    "",
    "## 5. Missing-Slot Age Intervention",
    "",
    "Age values for Mask=0 slots were replaced "
    "with the training-domain on-time Age median.",
    "",
    "This intervention is diagnostic only.",
    "",
    "| Direction | Variant | Mean abs. probability change | Prediction flips |",
    "|---|---|---:|---:|",
])

for _, row in shift.iterrows():
    lines.append(
        f"| {row['direction']} "
        f"| {row['variant']} "
        f"| {row['mean_abs_probability_shift']:.6f} "
        f"| {int(row['prediction_flips'])} |"
    )

lines.extend([
    "",
    "## 6. Attack Sequence 21–30",
    "",
    "| Direction | Variant | Original detected | Modified detected | N |",
    "|---|---|---:|---:|---:|",
])

for _, row in recovery.iterrows():
    lines.append(
        f"| {row['direction']} "
        f"| {row['variant']} "
        f"| {int(row['detected_original'])} "
        f"| {int(row['detected_modified'])} "
        f"| {int(row['n'])} |"
    )

lines.extend([
    "",
    "## 7. Methodological Interpretation",
    "",
    "This study reverses the monitoring condition "
    "and protocol bundle used for training and holdout.",
    "",
    "It does not isolate the causal effect of congestion.",
    "",
    "The training and holdout runs are exchanged "
    "between directions and are not independent replications.",
    "",
    "All comparisons are descriptive diagnostics.",
    "",
    "## 8. Limitations",
    "",
    "- Only one independent IFA run per domain.",
    "- M0 uses protocol v8 and M1 uses v16.",
    "- Overlapping temporal windows are correlated.",
    "- Three seeds do not provide independent network replications.",
    "- Both directions reuse previously examined development runs.",
    "- Plain can infer missingness from zero placeholders.",
    "- Age Valid is constant on these runs.",
    "- Intervention values can create unrealistic input combinations.",
    "- No model selection was performed using holdout metrics.",
    "",
    "## 9. Next Step",
    "",
    "Review the reciprocal evidence before freezing "
    "an inference model. Collect independent controlled "
    "evaluation runs before making robustness claims.",
    "",
])

path = OUT / "ml5d_report.md"
path.write_text(
    "\n".join(lines),
    encoding="utf-8"
)

print("Report created:", path)
