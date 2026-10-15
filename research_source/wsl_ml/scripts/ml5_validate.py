#!/usr/bin/env python3

import hashlib
import json
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

OUT = ROOT / "results/ml5"
CKPT = ROOT / "checkpoints/ml5"
DATA = ROOT / "data/processed/ml2"

config = json.loads(
    (ROOT / "configs/ml5/ablation.json").read_text()
)

manifest = json.loads(
    (OUT / "checkpoints.json").read_text()
)

predictions = pd.read_csv(
    OUT / "all_predictions.csv"
)

history = pd.read_csv(
    OUT / "training_history.csv"
)

encoder_path = (
    ROOT / "results/ml4/encoder.json"
)

encoder = MaskAgeEncoder.load(
    encoder_path
)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

dataset = NDNTemporalGraphDataset(
    data_dir=DATA,
    run_ids=config["holdout_runs"],
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
)

edges = dataset.edge_index.to(device)

assert len(manifest) == 9
assert len(predictions) == 9 * 276
assert len(history) == 9 * 20

print("=== ML-5 CHECKPOINT VALIDATION ===")

for item in manifest:
    seed = item["seed"]
    variant = item["variant"]
    path = ROOT / item["path"]

    assert path.is_file()

    actual_hash = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    assert actual_hash == item["sha256"]

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True
    )

    assert checkpoint["variant"] == variant
    assert checkpoint["seed"] == seed
    assert checkpoint["epoch"] == config["epochs"]
    assert checkpoint["train_runs"] == config["train_runs"]

    assert checkpoint["encoder_sha256"] == (
        hashlib.sha256(
            encoder_path.read_bytes()
        ).hexdigest()
    )

    model = GraphSAGEGRU(
        **checkpoint["model_config"]
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    reproduced = []

    with torch.no_grad():
        for batch in loader:
            gpu_batch = {
                k: v.to(device)
                if isinstance(v, torch.Tensor)
                else v
                for k, v in batch.items()
            }

            x = encoder.encode(
                gpu_batch, variant
            )

            probabilities = torch.sigmoid(
                model(x, edges)
            ).cpu().numpy()

            labels = batch["y"].numpy()
            sequences = batch[
                "window_end_sequence"
            ].numpy()

            for i in range(len(probabilities)):
                reproduced.append({
                    "run_id": batch["run_id"][i],
                    "end_sequence": int(sequences[i]),
                    "label": int(labels[i]),
                    "probability": float(
                        probabilities[i]
                    ),
                })

    reference = predictions[
        (predictions["seed"] == seed)
        & (predictions["variant"] == variant)
        & (predictions["split"] == "holdout")
    ].sort_values(
        ["run_id", "end_sequence"]
    ).reset_index(drop=True)

    new = pd.DataFrame(
        reproduced
    ).sort_values(
        ["run_id", "end_sequence"]
    ).reset_index(drop=True)

    assert len(reference) == 92
    assert len(new) == 92

    assert (
        reference["run_id"].tolist()
        == new["run_id"].tolist()
    )

    np.testing.assert_array_equal(
        reference["end_sequence"].to_numpy(),
        new["end_sequence"].to_numpy(),
    )

    np.testing.assert_array_equal(
        reference["label"].to_numpy(),
        new["label"].to_numpy(),
    )

    np.testing.assert_allclose(
        reference["probability"].to_numpy(),
        new["probability"].to_numpy(),
        rtol=1e-5,
        atol=1e-5,
    )

    assert sum(
        p.numel()
        for p in model.parameters()
    ) == 35009

    print(
        f"[PASS] {variant} seed={seed}: "
        "hash, architecture, "
        "and 92 predictions reproduced"
    )

# All models must use the same ground-truth labels.
holdout = predictions[
    predictions["split"] == "holdout"
]

reference_labels = None

for (seed, variant), group in holdout.groupby(
    ["seed", "variant"]
):
    group = group.sort_values(
        ["run_id", "end_sequence"]
    )
    labels = group["label"].to_numpy()

    if reference_labels is None:
        reference_labels = labels
    else:
        np.testing.assert_array_equal(
            reference_labels, labels
        )

print("[PASS] Identical holdout labels")

# All metrics must use the predefined threshold.
assert config["threshold"] == 0.5
assert config["checkpoint_selection"] == "final_epoch"

print("[PASS] Predefined evaluation policy")

result = {
    "status": "PASSED",
    "checkpoints": len(manifest),
    "reproduced_predictions_per_checkpoint": 92,
    "training_history_rows": len(history),
    "total_prediction_rows": len(predictions),
    "device": str(device),
}

(OUT / "ml5_validation.json").write_text(
    json.dumps(result, indent=2)
)

print("\n============================")
print("ML-5 VALIDATION: PASSED")
print("Checkpoints:", len(manifest))
print("Device:", device)
print("============================")
