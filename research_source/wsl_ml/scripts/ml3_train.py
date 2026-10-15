#!/usr/bin/env python3

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from temporal_dataset import NDNTemporalGraphDataset
from ml3_preprocessing import TrainOnlyScaler
from models.graphsage_gru import GraphSAGEGRU


CONFIG = ROOT / "configs/ml3/smoke.json"
DATA = ROOT / "data/processed/ml2"
OUT = ROOT / "results/ml3"
CKPT = ROOT / "checkpoints/ml3"

OUT.mkdir(parents=True, exist_ok=True)
CKPT.mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_metrics(y_true, probabilities, threshold):
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(
        probabilities, dtype=np.float64
    )

    predicted = (
        probabilities >= threshold
    ).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predicted,
        labels=[0, 1]
    ).ravel()

    result = {
        "samples": int(len(y_true)),
        "positives": int(y_true.sum()),
        "negatives": int(len(y_true) - y_true.sum()),
        "accuracy": float(
            accuracy_score(y_true, predicted)
        ),
        "precision": float(
            precision_score(
                y_true, predicted, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, predicted, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true, predicted, zero_division=0
            )
        ),
        "fpr": (
            float(fp / (fp + tn))
            if fp + tn > 0 else None
        ),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    if len(np.unique(y_true)) == 2:
        result["pr_auc"] = float(
            average_precision_score(
                y_true, probabilities
            )
        )
        result["roc_auc"] = float(
            roc_auc_score(
                y_true, probabilities
            )
        )
    else:
        result["pr_auc"] = None
        result["roc_auc"] = None

    return result


def run_inference(model, loader, scaler, device, edges):
    model.eval()

    labels = []
    probabilities = []
    prediction_rows = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            valid = batch["feature_valid"].to(device)
            y = batch["y"].to(device)

            x = scaler.transform(x, valid)

            logits = model(x, edges)
            probs = torch.sigmoid(logits)

            p_values = probs.cpu().numpy().tolist()
            y_values = y.cpu().numpy().tolist()

            labels.extend(y_values)
            probabilities.extend(p_values)

            for i in range(len(y_values)):
                prediction_rows.append({
                    "run_id": batch["run_id"][i],
                    "end_sequence": int(
                        batch["window_end_sequence"][i]
                    ),
                    "label": int(y_values[i]),
                    "probability": float(p_values[i]),
                })

    return labels, probabilities, prediction_rows


def main():
    config = json.loads(CONFIG.read_text())

    set_seed(config["seed"])

    train_runs = config["train_runs"]
    holdout_runs = config["diagnostic_holdout_runs"]

    assert set(train_runs).isdisjoint(holdout_runs)
    assert len(set(train_runs)) == len(train_runs)
    assert len(set(holdout_runs)) == len(holdout_runs)

    all_files = {
        p.stem for p in DATA.glob("*.npz")
    }

    assert set(train_runs + holdout_runs) == all_files

    train_ds = NDNTemporalGraphDataset(
        data_dir=DATA,
        run_ids=train_runs
    )

    holdout_ds = NDNTemporalGraphDataset(
        data_dir=DATA,
        run_ids=holdout_runs
    )

    assert len(train_ds) == 184
    assert len(holdout_ds) == 92

    assert torch.equal(
        train_ds.edge_index,
        holdout_ds.edge_index
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    edges = train_ds.edge_index.to(device)

    print("=== ML-3 CONFIGURATION ===")
    print("Device:", device)
    print("Training runs:", train_runs)
    print("Holdout runs:", holdout_runs)
    print("Training windows:", len(train_ds))
    print("Holdout windows:", len(holdout_ds))

    scaler = TrainOnlyScaler().fit(
        DATA, train_runs
    )

    scaler_path = OUT / "scaler.json"
    scaler.save(scaler_path)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0
    )

    train_eval_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0
    )

    holdout_loader = DataLoader(
        holdout_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0
    )

    model = GraphSAGEGRU(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        gru_hidden_dim=config["gru_hidden_dim"],
        dropout=config["dropout"],
    ).to(device)

    # Class weighting computed only from training labels.
    train_labels = np.array([
        int(sample["y"])
        for sample in train_ds.samples
    ])

    positive = int(train_labels.sum())
    negative = int(len(train_labels) - positive)

    if positive == 0 or negative == 0:
        raise ValueError(
            "Training needs both Normal and IFA labels"
        )

    pos_weight = negative / positive

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [pos_weight],
            device=device,
            dtype=torch.float32
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    history = []

    print("\n=== TRAINING ===")
    print("Class weights:", {
        "positive": positive,
        "negative": negative,
        "pos_weight": pos_weight
    })

    for epoch in range(1, config["epochs"] + 1):
        model.train()

        total_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            valid = batch["feature_valid"].to(device)
            y = batch["y"].float().to(device)

            x = scaler.transform(x, valid)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x, edges)
            loss = criterion(logits, y)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Nonfinite training loss"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            batch_size = x.shape[0]
            total_samples += batch_size
            total_loss += (
                loss.item() * batch_size
            )

        mean_loss = total_loss / total_samples

        history.append({
            "epoch": epoch,
            "train_loss": mean_loss
        })

        print(
            f"Epoch {epoch:02d}/{config['epochs']} "
            f"Train Loss: {mean_loss:.6f}"
        )

    # Final epoch checkpoint; holdout has not been consulted.
    checkpoint_path = CKPT / "graphsage_gru_plain.pt"

    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": config["input_dim"],
            "hidden_dim": config["hidden_dim"],
            "gru_hidden_dim": config["gru_hidden_dim"],
            "dropout": config["dropout"],
        },
        "epoch": config["epochs"],
    }, checkpoint_path)

    with (OUT / "training_history.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss"]
        )
        writer.writeheader()
        writer.writerows(history)

    # Final diagnostic evaluation.
    # No hyperparameter or epoch selection from holdout.
    train_y, train_p, _ = run_inference(
        model, train_eval_loader,
        scaler, device, edges
    )

    holdout_y, holdout_p, predictions = run_inference(
        model, holdout_loader,
        scaler, device, edges
    )

    threshold = config["threshold"]

    train_metrics = calculate_metrics(
        train_y, train_p, threshold
    )

    holdout_metrics = calculate_metrics(
        holdout_y, holdout_p, threshold
    )

    per_run = {}

    for run_id in holdout_runs:
        selected = [
            row for row in predictions
            if row["run_id"] == run_id
        ]

        per_run[run_id] = calculate_metrics(
            [r["label"] for r in selected],
            [r["probability"] for r in selected],
            threshold
        )

    with (OUT / "holdout_predictions.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "end_sequence",
                "label",
                "probability"
            ]
        )
        writer.writeheader()
        writer.writerows(predictions)

    report = {
        "experiment": config["experiment"],
        "purpose": config["purpose"],
        "training_runs": train_runs,
        "diagnostic_holdout_runs": holdout_runs,
        "training_windows": len(train_ds),
        "holdout_windows": len(holdout_ds),
        "train_positives": positive,
        "train_negatives": negative,
        "pos_weight": pos_weight,
        "checkpoint_selection": "final_epoch",
        "threshold": threshold,
        "train_metrics": train_metrics,
        "diagnostic_holdout_metrics": holdout_metrics,
        "diagnostic_holdout_per_run": per_run,
        "limitations": [
            "Only two independent IFA runs exist.",
            "Train uses v8/v14; holdout uses v16.",
            "Overlapping windows are not independent.",
            "No model or threshold selection on holdout.",
            "Metrics are diagnostic, not robustness claims.",
        ],
    }

    (OUT / "training_report.json").write_text(
        json.dumps(report, indent=2)
    )

    (OUT / "experiment_config.json").write_text(
        json.dumps(config, indent=2)
    )

    print("\n=== FINAL RESULTS ===")
    print("Checkpoint:", checkpoint_path)
    print("Scaler:", scaler_path)
    print("Train metrics:")
    print(json.dumps(train_metrics, indent=2))
    print("Diagnostic holdout metrics:")
    print(json.dumps(holdout_metrics, indent=2))
    print("\nML-3 TRAINING COMPLETED")


if __name__ == "__main__":
    main()
