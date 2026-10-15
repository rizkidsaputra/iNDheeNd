#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from temporal_dataset import NDNTemporalGraphDataset
from ml3_preprocessing import TrainOnlyScaler
from models.graphsage_gru import GraphSAGEGRU


CONFIG = ROOT / "configs/ml3/smoke.json"
DATA = ROOT / "data/processed/ml2"
OUT = ROOT / "results/ml3"
CKPT = ROOT / "checkpoints/ml3/graphsage_gru_plain.pt"


def make_model(config, device):
    return GraphSAGEGRU(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        gru_hidden_dim=config["gru_hidden_dim"],
        dropout=config["dropout"]
    ).to(device)


def main():
    config = json.loads(CONFIG.read_text())

    train_ids = config["train_runs"]
    holdout_ids = config["diagnostic_holdout_runs"]

    assert set(train_ids).isdisjoint(holdout_ids)
    print("[PASS] Run-level split disjoint")

    train_ds = NDNTemporalGraphDataset(
        data_dir=DATA, run_ids=train_ids
    )

    holdout_ds = NDNTemporalGraphDataset(
        data_dir=DATA, run_ids=holdout_ids
    )

    assert len(train_ds) == 184
    assert len(holdout_ds) == 92

    print("[PASS] Dataset split sizes")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    edges = train_ds.edge_index.to(device)

    train_batch = next(iter(DataLoader(
        train_ds,
        batch_size=4,
        shuffle=False
    )))

    scaler = TrainOnlyScaler.load(
        OUT / "scaler.json"
    )

    assert scaler.train_runs == train_ids
    print("[PASS] Scaler training-run provenance")

    # Independently refit scaler from train runs and compare.
    independent = TrainOnlyScaler().fit(
        DATA, train_ids
    )

    np.testing.assert_allclose(
        independent.mean,
        scaler.mean,
        rtol=1e-7
    )
    np.testing.assert_allclose(
        independent.std,
        scaler.std,
        rtol=1e-7
    )

    print("[PASS] Training-only scaler reproducibility")

    x = train_batch["x"].to(device)
    valid = train_batch["feature_valid"].to(device)

    scaled = scaler.transform(x, valid)

    assert scaled.shape == (4, 5, 10, 10)
    assert torch.isfinite(scaled).all()

    assert torch.all(
        scaled[valid == 0] == 0
    )

    print("[PASS] Missing-value placeholder semantics")

    # Forward and backward test.
    model = make_model(config, device)
    model.train()

    logits = model(scaled, edges)

    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()

    target = train_batch["y"].float().to(device)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target
    )

    loss.backward()

    gradients = [
        p.grad
        for p in model.parameters()
        if p.requires_grad
    ]

    assert all(g is not None for g in gradients)
    assert all(
        torch.isfinite(g).all()
        for g in gradients
    )

    print("[PASS] GraphSAGE + GRU forward/backward")

    assert CKPT.is_file()

    saved = torch.load(
        CKPT,
        map_location=device,
        weights_only=True
    )

    trained = make_model(config, device)
    trained.load_state_dict(
        saved["model_state_dict"]
    )

    trained.eval()

    with torch.no_grad():
        p1 = trained.predict_proba(
            scaled, edges
        )

    # Reload independently and compare predictions.
    reloaded = make_model(config, device)
    reloaded.load_state_dict(
        torch.load(
            CKPT,
            map_location=device,
            weights_only=True
        )["model_state_dict"]
    )

    reloaded.eval()

    with torch.no_grad():
        p2 = reloaded.predict_proba(
            scaled, edges
        )

    torch.testing.assert_close(
        p1, p2,
        rtol=1e-6,
        atol=1e-6
    )

    assert ((p1 >= 0) & (p1 <= 1)).all()

    print("[PASS] Checkpoint reload consistency")

    report = json.loads(
        (OUT / "training_report.json").read_text()
    )

    assert report["training_windows"] == 184
    assert report["holdout_windows"] == 92
    assert report["train_positives"] == 15

    print("[PASS] Training report consistency")

    print("\n==========================")
    print("ML-3 VALIDATION: PASSED")
    print("Device:", device)
    print("Train windows:", len(train_ds))
    print("Holdout windows:", len(holdout_ds))
    print("Model output:", tuple(p1.shape))
    print("==========================")


if __name__ == "__main__":
    main()
