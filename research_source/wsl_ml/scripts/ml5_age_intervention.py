import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from temporal_dataset import NDNTemporalGraphDataset
from ml4_observation import MaskAgeEncoder
from models.graphsage_gru import GraphSAGEGRU

DATA = ROOT / "data/processed/ml2"
CKPT = ROOT / "checkpoints/ml5"
OUT = ROOT / "results/ml5"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

encoder = MaskAgeEncoder.load(
    ROOT / "results/ml4/encoder.json"
)

dataset = NDNTemporalGraphDataset(
    data_dir=DATA,
    run_ids=["m1dev_v16_r02_a1"]
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False
)

edges = dataset.edge_index.to(device)

conditions = [
    "original",
    "nominal_all",
    "nominal_missing",
    "clip_6s",
]

rows = []

for seed in [42, 52, 62]:
    checkpoint = torch.load(
        CKPT / f"mask_age_seed{seed}.pt",
        map_location=device,
        weights_only=True
    )

    model = GraphSAGEGRU(
        **checkpoint["model_config"]
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    with torch.no_grad():
        for batch in loader:
            b = {
                k: v.to(device)
                if isinstance(v, torch.Tensor)
                else v
                for k, v in batch.items()
            }

            for condition in conditions:
                modified = dict(b)

                age = b["age"].clone()
                valid = b["age_valid"] > 0.5
                mask = b["mask"]

                if condition == "nominal_all":
                    age = torch.where(
                        valid,
                        torch.full_like(age, 2.34),
                        age
                    )

                elif condition == "nominal_missing":
                    age = torch.where(
                        valid & (mask == 0),
                        torch.full_like(age, 2.34),
                        age
                    )

                elif condition == "clip_6s":
                    age = torch.where(
                        valid,
                        age.clamp(max=6.0),
                        age
                    )

                modified["age"] = age

                x = encoder.encode(
                    modified, "mask_age"
                )

                p = torch.sigmoid(
                    model(x, edges)
                ).cpu().numpy()

                labels = batch["y"].numpy()
                sequences = batch[
                    "window_end_sequence"
                ].numpy()

                for i in range(len(p)):
                    rows.append({
                        "seed": seed,
                        "condition": condition,
                        "sequence": int(sequences[i]),
                        "label": int(labels[i]),
                        "probability": float(p[i]),
                        "predicted": int(p[i] >= 0.5),
                    })

df = pd.DataFrame(rows)

path = OUT / "age_intervention_diagnostic.csv"
df.to_csv(path, index=False)

print("=== AGE INTERVENTION RESULTS ===")

for condition in conditions:
    subset = df[
        (df["condition"] == condition)
        & (df["label"] == 1)
    ]

    for seed in [42, 52, 62]:
        s = subset[subset["seed"] == seed]

        late_phase = s[
            s["sequence"].between(21, 30)
        ]

        print({
            "condition": condition,
            "seed": seed,
            "IFA_detected": int(
                s["predicted"].sum()
            ),
            "IFA_total": len(s),
            "seq21_30_detected": int(
                late_phase["predicted"].sum()
            ),
            "seq21_30_total": len(late_phase),
            "seq21_30_mean_probability": round(
                float(late_phase["probability"].mean()),
                4
            ),
        })

print("\n=== PER-SEQUENCE MEAN ACROSS SEEDS ===")

table = df[
    df["sequence"].between(16, 31)
].pivot_table(
    index="sequence",
    columns="condition",
    values="probability",
    aggfunc="mean"
)

print(table.round(4).to_string())

print("\nSaved:", path)
print("ML-5 Age intervention completed")
