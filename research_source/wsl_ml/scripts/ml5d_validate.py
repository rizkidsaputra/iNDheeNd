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
from ml3_preprocessing import TrainOnlyScaler
from ml4_observation import MaskAgeEncoder
from models.graphsage_gru import GraphSAGEGRU

CONFIG = json.loads(
    (ROOT / "configs/ml5d/reciprocal.json").read_text()
)

OUT = ROOT / "results/ml5d"
DATA = ROOT / "data/processed/ml2"

manifest = json.loads(
    (OUT / "checkpoints.json").read_text()
)

audit = json.loads(
    (OUT / "split_and_scaler_audit.json").read_text()
)

saved = pd.read_csv(
    OUT / "all_predictions.csv"
)

history = pd.read_csv(
    OUT / "training_history.csv"
)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


def sha256(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def gpu(batch):
    return {
        k: v.to(device)
        if isinstance(v, torch.Tensor)
        else v
        for k, v in batch.items()
    }


def reproduce(model, encoder, loader, edges, variant):
    rows = []
    model.eval()

    with torch.no_grad():
        for batch in loader:
            b = gpu(batch)

            x = encoder.encode(b, variant)

            probabilities = torch.sigmoid(
                model(x, edges)
            ).cpu().numpy()

            for i, probability in enumerate(probabilities):
                rows.append({
                    "run_id": batch["run_id"][i],
                    "end_sequence": int(
                        batch["window_end_sequence"][i]
                    ),
                    "label": int(batch["y"][i]),
                    "probability": float(probability),
                })

    return pd.DataFrame(rows).sort_values(
        ["run_id", "end_sequence"]
    ).reset_index(drop=True)


assert len(manifest) == 18
assert len(audit) == 2
assert len(history) == 360

# 18 models x (92 training original +
# 92 holdout original + 92 holdout intervention).
assert len(saved) == 18 * 276

print("=== 1. RECIPROCAL SPLITS AND SCALERS ===")

for entry in audit:
    direction = entry["direction"]

    train = entry["train_runs"]
    holdout = entry["holdout_runs"]

    assert set(train).isdisjoint(holdout)

    assert entry["training_windows"] == 92
    assert entry["holdout_windows"] == 92
    assert entry["training_positives"] == 15
    assert entry["holdout_positives"] == 15

    folder = OUT / direction

    feature_path = folder / "feature_scaler.json"
    encoder_path = folder / "encoder.json"

    assert sha256(feature_path) == (
        entry["feature_scaler_sha256"]
    )

    assert sha256(encoder_path) == (
        entry["encoder_sha256"]
    )

    feature_scaler = TrainOnlyScaler.load(
        feature_path
    )

    assert feature_scaler.train_runs == train

    independently_fitted = TrainOnlyScaler().fit(
        DATA, train
    )

    np.testing.assert_allclose(
        feature_scaler.mean,
        independently_fitted.mean,
        rtol=1e-7
    )

    np.testing.assert_allclose(
        feature_scaler.std,
        independently_fitted.std,
        rtol=1e-7
    )

    encoder = MaskAgeEncoder.load(
        encoder_path
    )

    assert encoder.train_runs == train

    independent_encoder = MaskAgeEncoder.fit(
        DATA, train, feature_path
    )

    np.testing.assert_allclose(
        encoder.age_mean,
        independent_encoder.age_mean,
        rtol=1e-7
    )

    np.testing.assert_allclose(
        encoder.age_std,
        independent_encoder.age_std,
        rtol=1e-7
    )

    print(
        "[PASS]", direction,
        "run-disjoint and train-only scalers"
    )


print("\n=== 2. CHECKPOINT REPRODUCTION ===")

for item in manifest:
    direction = item["direction"]
    variant = item["variant"]
    seed = item["seed"]

    path = ROOT / item["checkpoint"]
    assert sha256(path) == item["sha256"]

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True
    )

    assert checkpoint["direction"] == direction
    assert checkpoint["variant"] == variant
    assert checkpoint["seed"] == seed
    assert checkpoint["epoch"] == CONFIG["epochs"]

    split = CONFIG["directions"][direction]

    assert (
        checkpoint["train_runs"]
        == split["train_runs"]
    )

    assert (
        checkpoint["holdout_runs"]
        == split["holdout_runs"]
    )

    encoder_path = (
        OUT / direction / "encoder.json"
    )

    assert checkpoint["encoder_sha256"] == (
        sha256(encoder_path)
    )

    encoder = MaskAgeEncoder.load(
        encoder_path
    )

    ds = NDNTemporalGraphDataset(
        data_dir=DATA,
        run_ids=split["holdout_runs"]
    )

    loader = DataLoader(
        ds,
        batch_size=16,
        shuffle=False
    )

    edges = ds.edge_index.to(device)

    model = GraphSAGEGRU(
        **checkpoint["model_config"]
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    assert sum(
        p.numel() for p in model.parameters()
    ) == 35009

    actual = reproduce(
        model, encoder, loader, edges, variant
    )

    reference = saved[
        (saved["direction"] == direction)
        & (saved["seed"] == seed)
        & (saved["variant"] == variant)
        & (saved["split"] == "holdout")
        & (saved["condition"] == "original")
    ].sort_values(
        ["run_id", "end_sequence"]
    ).reset_index(drop=True)

    assert len(actual) == len(reference) == 92

    assert (
        actual["run_id"].tolist()
        == reference["run_id"].tolist()
    )

    np.testing.assert_array_equal(
        actual["end_sequence"],
        reference["end_sequence"]
    )

    np.testing.assert_array_equal(
        actual["label"],
        reference["label"]
    )

    np.testing.assert_allclose(
        actual["probability"],
        reference["probability"],
        rtol=1e-5,
        atol=1e-5,
    )

    print(
        f"[PASS] {direction}, "
        f"{variant}, seed={seed}: "
        "92 predictions reproduced"
    )


print("\n=== 3. PAIRED EXPERIMENT CHECK ===")

# For each direction+seed, all variants must have
# identical initialization and ground-truth labels.
for direction in CONFIG["directions"]:
    for seed in CONFIG["seeds"]:
        selected = [
            x for x in manifest
            if x["direction"] == direction
            and x["seed"] == seed
        ]

        assert len(selected) == 3

        signatures = {
            x["initialization_sha256"]
            for x in selected
        }

        assert len(signatures) == 1

        reference_labels = None

        for variant in CONFIG["variants"]:
            rows = saved[
                (saved["direction"] == direction)
                & (saved["seed"] == seed)
                & (saved["variant"] == variant)
                & (saved["split"] == "holdout")
                & (saved["condition"] == "original")
            ].sort_values(
                ["run_id", "end_sequence"]
            )

            labels = rows["label"].to_numpy()

            if reference_labels is None:
                reference_labels = labels
            else:
                np.testing.assert_array_equal(
                    reference_labels, labels
                )

        print(
            "[PASS]",
            direction,
            "seed",
            seed,
            "initialization and labels matched"
        )


result = {
    "status": "PASSED",
    "directions": 2,
    "models": 18,
    "predictions_reproduced_per_model": 92,
    "training_history_rows": len(history),
    "all_prediction_rows": len(saved),
    "device": str(device),
    "note": "Diagnostic experiment, not independent robustness evidence"
}

(OUT / "ml5d_validation.json").write_text(
    json.dumps(result, indent=2)
)

print("\n============================")
print("ML-5D VALIDATION: PASSED")
print("Directions: 2")
print("Checkpoints: 18")
print("Device:", device)
print("============================")
