#!/usr/bin/env python3

import json
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / "results/ml5"

summary = json.loads(
    (OUT / "ablation_summary.json").read_text()
)
validation = json.loads(
    (OUT / "ml5_validation.json").read_text()
)
sensitivity = json.loads(
    (OUT / "sensitivity.json").read_text()
)
probes = json.loads(
    (OUT / "shortcut_probes.json").read_text()
)

metrics = pd.read_csv(
    OUT / "metrics_by_run.csv"
)

boundary = pd.read_csv(
    OUT / "boundary_predictions.csv"
)

assert validation["status"] == "PASSED"
assert summary["models_trained"] == 9

lines = [
    "# ML-5 — Controlled Training, Ablation & Diagnostics",
    "",
    "## 1. Experiment Status",
    "",
    "**Implementation and checkpoint validation: PASSED**",
    "",
    "- Models trained: 9",
    "- Variants: Plain, Mask, Mask+Age",
    "- Seeds: 42, 52, 62",
    "- Epochs per model: 20",
    "- Parameters per model: 35,009",
    "- Training windows per model: 184",
    "- Diagnostic holdout windows per model: 92",
    "- Checkpoint: final epoch",
    "- Decision threshold: 0.5",
    "",
    "## 2. Descriptive Holdout Comparison",
    "",
    "| Variant | F1 mean | Recall mean | FPR mean | AP mean |",
    "|---|---:|---:|---:|---:|",
]

for item in summary["holdout_aggregate"]:
    lines.append(
        f"| {item['variant']} "
        f"| {item['f1_mean']:.4f} "
        f"| {item['recall_mean']:.4f} "
        f"| {item['fpr_mean']:.4f} "
        f"| {item['average_precision_mean']:.4f} |"
    )

lines.extend([
    "",
    "Means are descriptive averages over three seeds, "
    "not independent experimental replications.",
    "",
    "## 3. Individual Seed Results",
    "",
    "| Seed | Variant | F1 | Recall | FPR | AP |",
    "|---:|---|---:|---:|---:|---:|",
])

rows = metrics[
    (metrics["split"] == "holdout")
    & (metrics["run_id"] == "ALL")
].sort_values(
    ["seed", "variant"]
)

for _, r in rows.iterrows():
    lines.append(
        f"| {int(r['seed'])} "
        f"| {r['variant']} "
        f"| {r['f1']:.4f} "
        f"| {r['recall']:.4f} "
        f"| {r['fpr']:.4f} "
        f"| {r['average_precision']:.4f} |"
    )

lines.extend([
    "",
    "## 4. Attack Boundary Diagnostics",
    "",
    "Examined attack start/end transition sequences: "
    "16, 17, 30, and 31.",
    "",
    "| Seed | Variant | Seq 16 | Seq 31 |",
    "|---:|---|---:|---:|",
])

for (seed, variant), group in boundary.groupby(
    ["seed", "variant"]
):
    by_seq = {
        int(r["end_sequence"]): r
        for _, r in group.iterrows()
    }

    lines.append(
        f"| {int(seed)} | {variant} "
        f"| {by_seq[16]['probability']:.4f} "
        f"| {by_seq[31]['probability']:.4f} |"
    )

lines.extend([
    "",
    "## 5. Channel Intervention Sensitivity",
    "",
    "| Seed | Variant | Mean absolute probability change | Flips |",
    "|---:|---|---:|---:|",
])

for item in sensitivity:
    lines.append(
        f"| {item['seed']} "
        f"| {item['variant']} "
        f"| {item['mean_abs_delta_probability']:.6f} "
        f"| {item['prediction_flips']} |"
    )

lines.extend([
    "",
    "Channel interventions are diagnostic only. "
    "They are not estimates of causal improvement.",
    "",
    "## 6. Observation-Only Shortcut Probes",
    "",
    "| Probe | F1 | Recall | AP |",
    "|---|---:|---:|---:|",
])

for item in probes:
    lines.append(
        f"| {item['probe']} "
        f"| {item['f1']:.4f} "
        f"| {item['recall']:.4f} "
        f"| {item['average_precision']:.4f} |"
    )

lines.extend([
    "",
    "## 7. Methodological Limitations",
    "",
    "- Only one independent IFA run is used in training.",
    "- Only one independent IFA run is used in holdout.",
    "- Training protocols are v8/v14 and holdout is v16.",
    "- The same diagnostic holdout was previously examined in ML-3.",
    "- Overlapping temporal windows are correlated.",
    "- Three random seeds do not constitute three independent network runs.",
    "- The Plain model may infer missingness from feature placeholders.",
    "- Age Valid is constant in the current dataset.",
    "- High diagnostic F1 does not establish robustness.",
    "",
    "## 8. Conclusion Scope",
    "",
    "ML-5 evaluates whether the three implementations "
    "can be trained and compared under identical "
    "computational settings. The resulting differences "
    "are preliminary and must be verified with "
    "additional independent experiments.",
    "",
    "## 9. Next Milestone",
    "",
    "ML-6 — Offline Inference & Model Packaging.",
    "",
])

path = OUT / "ml5_report.md"
path.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print("Report saved:", path)
