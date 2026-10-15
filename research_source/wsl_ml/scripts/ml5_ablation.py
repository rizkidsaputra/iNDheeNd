#!/usr/bin/env python3

import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
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
from ml4_observation import MaskAgeEncoder
from models.graphsage_gru import GraphSAGEGRU

CONFIG_PATH = ROOT / "configs/ml5/ablation.json"
DATA = ROOT / "data/processed/ml2"
OUT = ROOT / "results/ml5"
CKPT = ROOT / "checkpoints/ml5"

OUT.mkdir(parents=True, exist_ok=True)
CKPT.mkdir(parents=True, exist_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def write_csv(path, rows, columns):
    with Path(path).open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=columns
        )
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def metrics(rows, threshold):
    y = np.asarray(
        [r["label"] for r in rows],
        dtype=np.int64,
    )
    p = np.asarray(
        [r["probability"] for r in rows],
        dtype=np.float64,
    )

    assert len(y) > 0
    assert np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()

    pred = (p >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    has_pos = bool(np.any(y == 1))
    has_neg = bool(np.any(y == 0))

    result = {
        "n": int(len(y)),
        "positive": int(y.sum()),
        "negative": int((y == 0).sum()),
        "accuracy": float(
            accuracy_score(y, pred)
        ),
        "precision": (
            float(precision_score(y, pred))
            if has_pos and np.any(pred == 1)
            else (0.0 if has_pos else None)
        ),
        "recall": (
            float(recall_score(y, pred))
            if has_pos else None
        ),
        "f1": (
            float(f1_score(y, pred, zero_division=0))
            if has_pos else None
        ),
        "fpr": (
            float(fp / (fp + tn))
            if has_neg else None
        ),
        "average_precision": (
            float(average_precision_score(y, p))
            if has_pos and has_neg else None
        ),
        "roc_auc": (
            float(roc_auc_score(y, p))
            if has_pos and has_neg else None
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return result


def move_batch(batch, device):
    return {
        k: v.to(device)
        if isinstance(v, torch.Tensor)
        else v
        for k, v in batch.items()
    }


def build_model(config):
    return GraphSAGEGRU(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        gru_hidden_dim=config["gru_hidden_dim"],
        dropout=config["dropout"],
    )


@torch.no_grad()
def predict(
    model, loader, encoder,
    variant, device, edges
):
    model.eval()
    rows = []

    for batch in loader:
        batch = move_batch(batch, device)
        x = encoder.encode(batch, variant)

        probabilities = torch.sigmoid(
            model(x, edges)
        ).cpu().numpy()

        labels = batch["y"].cpu().numpy()
        sequences = (
            batch["window_end_sequence"]
            .cpu().numpy()
        )

        for i in range(len(labels)):
            rows.append({
                "run_id": batch["run_id"][i],
                "end_sequence": int(sequences[i]),
                "label": int(labels[i]),
                "probability": float(
                    probabilities[i]
                ),
                "missing_at_decision": int(
                    (batch["mask"][i, -1] == 0)
                    .sum().item()
                ),
            })

    return rows


@torch.no_grad()
def sensitivity(
    model, loader, encoder,
    variant, device, edges,
):
    """
    Channel intervention diagnostic only.
    Not causal evidence of performance improvement.
    """
    model.eval()

    differences = []
    flips = 0
    total = 0

    if variant == "plain":
        return {
            "mean_abs_delta_probability": 0.0,
            "prediction_flips": 0,
            "samples": 0,
            "channel_intervention": "none",
        }

    channel = (
        10 if variant == "mask"
        else 11
    )

    for batch in loader:
        batch = move_batch(batch, device)
        x = encoder.encode(batch, variant)

        baseline = torch.sigmoid(
            model(x, edges)
        )

        modified = x.clone()

        # Mask: set its channel to zero.
        # Mask+Age: set only normalized Age to zero.
        # Other channels are unchanged.
        modified[..., channel] = 0

        altered = torch.sigmoid(
            model(modified, edges)
        )

        differences.extend(
            (baseline - altered)
            .abs().cpu().tolist()
        )

        flips += int(
            (
                (baseline >= 0.5)
                != (altered >= 0.5)
            ).sum().item()
        )

        total += len(baseline)

    return {
        "channel_intervention": (
            "mask_zeroed"
            if variant == "mask"
            else "standardized_age_zeroed"
        ),
        "mean_abs_delta_probability": float(
            np.mean(differences)
        ),
        "prediction_flips": flips,
        "samples": total,
        "interpretation": (
            "Model sensitivity under a channel "
            "intervention, not a causal estimate."
        ),
    }


def main():
    config = read_json(CONFIG_PATH)

    assert config["variants"] == [
        "plain", "mask", "mask_age"
    ]
    assert config["no_holdout_model_selection"]
    assert config["checkpoint_selection"] == "final_epoch"

    train_runs = config["train_runs"]
    holdout_runs = config["holdout_runs"]

    assert set(train_runs).isdisjoint(
        set(holdout_runs)
    )

    assert len(set(train_runs)) == len(train_runs)
    assert len(set(holdout_runs)) == len(holdout_runs)

    available = {
        p.stem for p in DATA.glob("*.npz")
    }
    assert set(train_runs + holdout_runs) == available

    ml4_validation = read_json(
        ROOT / "results/ml4/ml4_validation.json"
    )
    causal_audit = read_json(
        ROOT / "results/ml4/causal_audit.json"
    )

    assert ml4_validation["status"] == "PASSED"
    assert causal_audit["status"] == "PASSED"
    assert causal_audit["slot_checks"] == 3000

    encoder_path = (
        ROOT / "results/ml4/encoder.json"
    )
    encoder = MaskAgeEncoder.load(
        encoder_path
    )

    assert encoder.train_runs == train_runs

    train_ds = NDNTemporalGraphDataset(
        data_dir=DATA,
        run_ids=train_runs,
    )
    holdout_ds = NDNTemporalGraphDataset(
        data_dir=DATA,
        run_ids=holdout_runs,
    )

    assert len(train_ds) == 184
    assert len(holdout_ds) == 92
    assert torch.equal(
        train_ds.edge_index,
        holdout_ds.edge_index
    )

    labels = np.asarray(
        [s["y"] for s in train_ds.samples]
    )

    positives = int(labels.sum())
    negatives = int((labels == 0).sum())

    assert positives == 15
    assert negatives == 169

    pos_weight = negatives / positives

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    edges = train_ds.edge_index.to(device)

    threshold = float(config["threshold"])

    all_predictions = []
    metric_rows = []
    histories = []
    sensitivity_rows = []
    checkpoint_manifest = []

    print("=== ML-5 CONTROLLED ABLATION ===")
    print("Device:", device)
    print("Train windows:", len(train_ds))
    print("Holdout windows:", len(holdout_ds))
    print("Positive weight:", pos_weight)
    print("Variants:", config["variants"])
    print("Seeds:", config["seeds"])

    for seed in config["seeds"]:
        for variant in config["variants"]:
            print(
                f"\n=== SEED {seed} / "
                f"VARIANT {variant} ==="
            )

            # Reset RNG before each arm.
            # Identical seed gives identical
            # model initialization for each arm.
            set_seed(seed)

            model = build_model(config).to(
                device
            )

            parameters = sum(
                p.numel() for p in model.parameters()
            )
            assert parameters == 35009

            # Fixed batch-order generator for each arm.
            generator = torch.Generator()
            generator.manual_seed(seed)

            train_loader = DataLoader(
                train_ds,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=0,
                generator=generator,
            )

            train_eval_loader = DataLoader(
                train_ds,
                batch_size=config["batch_size"],
                shuffle=False,
            )

            holdout_loader = DataLoader(
                holdout_ds,
                batch_size=config["batch_size"],
                shuffle=False,
            )

            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(
                    pos_weight,
                    dtype=torch.float32,
                    device=device,
                )
            )

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"],
            )

            for epoch in range(
                1, config["epochs"] + 1
            ):
                model.train()
                running_loss = 0.0
                n = 0

                for batch in train_loader:
                    batch = move_batch(
                        batch, device
                    )

                    x = encoder.encode(
                        batch, variant
                    )
                    y = batch["y"].float()

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    logits = model(x, edges)
                    loss = criterion(logits, y)

                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            "Nonfinite training loss"
                        )

                    loss.backward()

                    nn.utils.clip_grad_norm_(
                        model.parameters(),
                        1.0
                    )

                    optimizer.step()

                    size = len(y)
                    n += size
                    running_loss += (
                        loss.item() * size
                    )

                epoch_loss = running_loss / n

                histories.append({
                    "seed": seed,
                    "variant": variant,
                    "epoch": epoch,
                    "train_loss": epoch_loss,
                })

                print(
                    f"Epoch {epoch:02d}/"
                    f"{config['epochs']} "
                    f"Loss={epoch_loss:.6f}"
                )

            # Save final epoch, never choose using holdout.
            checkpoint_path = CKPT / (
                f"{variant}_seed{seed}.pt"
            )

            torch.save({
                "model_state_dict":
                    model.state_dict(),
                "model_config": {
                    "input_dim": config["input_dim"],
                    "hidden_dim": config["hidden_dim"],
                    "gru_hidden_dim":
                        config["gru_hidden_dim"],
                    "dropout": config["dropout"],
                },
                "variant": variant,
                "seed": seed,
                "epoch": config["epochs"],
                "train_runs": train_runs,
                "encoder_sha256": sha256(
                    encoder_path
                ),
            }, checkpoint_path)

            checkpoint_manifest.append({
                "seed": seed,
                "variant": variant,
                "path": str(
                    checkpoint_path.relative_to(ROOT)
                ),
                "sha256": sha256(
                    checkpoint_path
                ),
                "parameters": parameters,
            })

            train_predictions = predict(
                model, train_eval_loader,
                encoder, variant, device, edges
            )

            holdout_predictions = predict(
                model, holdout_loader,
                encoder, variant, device, edges
            )

            for split, rows in [
                ("train", train_predictions),
                ("holdout", holdout_predictions),
            ]:
                for r in rows:
                    r["split"] = split
                    r["seed"] = seed
                    r["variant"] = variant
                    r["prediction"] = int(
                        r["probability"] >= threshold
                    )

                all_predictions.extend(rows)

                overall = metrics(
                    rows, threshold
                )

                metric_rows.append({
                    "seed": seed,
                    "variant": variant,
                    "split": split,
                    "run_id": "ALL",
                    **overall,
                })

                run_ids = (
                    train_runs if split == "train"
                    else holdout_runs
                )

                for run_id in run_ids:
                    selected = [
                        r for r in rows
                        if r["run_id"] == run_id
                    ]

                    metric_rows.append({
                        "seed": seed,
                        "variant": variant,
                        "split": split,
                        "run_id": run_id,
                        **metrics(selected, threshold),
                    })

            diagnostic = sensitivity(
                model,
                holdout_loader,
                encoder,
                variant,
                device,
                edges,
            )

            sensitivity_rows.append({
                "seed": seed,
                "variant": variant,
                **diagnostic,
            })

            print(
                "Train F1:",
                metrics(
                    train_predictions, threshold
                )["f1"]
            )
            print(
                "Holdout F1:",
                metrics(
                    holdout_predictions, threshold
                )["f1"]
            )

    write_csv(
        OUT / "all_predictions.csv",
        all_predictions,
        [
            "seed", "variant", "split",
            "run_id", "end_sequence",
            "label", "probability",
            "prediction", "missing_at_decision",
        ]
    )

    metric_columns = [
        "seed", "variant", "split", "run_id",
        "n", "positive", "negative",
        "accuracy", "precision", "recall",
        "f1", "fpr", "average_precision",
        "roc_auc", "tn", "fp", "fn", "tp",
    ]

    write_csv(
        OUT / "metrics_by_run.csv",
        metric_rows,
        metric_columns
    )

    write_csv(
        OUT / "training_history.csv",
        histories,
        [
            "seed", "variant",
            "epoch", "train_loss"
        ]
    )

    (OUT / "sensitivity.json").write_text(
        json.dumps(
            sensitivity_rows, indent=2
        )
    )

    (OUT / "checkpoints.json").write_text(
        json.dumps(
            checkpoint_manifest, indent=2
        )
    )

    (OUT / "experiment_config.json").write_text(
        json.dumps(config, indent=2)
    )

    # Descriptive seed aggregation only.
    holdout_aggregate = []

    for variant in config["variants"]:
        selected = [
            r for r in metric_rows
            if r["variant"] == variant
            and r["split"] == "holdout"
            and r["run_id"] == "ALL"
        ]

        assert len(selected) == len(
            config["seeds"]
        )

        row = {"variant": variant}

        for name in [
            "accuracy", "precision",
            "recall", "f1", "fpr",
            "average_precision", "roc_auc",
        ]:
            values = [
                float(r[name])
                for r in selected
            ]

            row[f"{name}_mean"] = float(
                np.mean(values)
            )

            row[f"{name}_std"] = float(
                np.std(values)
            )

        holdout_aggregate.append(row)

    write_csv(
        OUT / "holdout_aggregate.csv",
        holdout_aggregate,
        list(holdout_aggregate[0].keys())
    )

    summary = {
        "status": "TRAINING_COMPLETED",
        "models_trained": len(
            checkpoint_manifest
        ),
        "seeds": config["seeds"],
        "variants": config["variants"],
        "train_windows_per_model": len(train_ds),
        "holdout_windows_per_model":
            len(holdout_ds),
        "parameters_per_model": 35009,
        "checkpoint_selection": "final_epoch",
        "threshold": threshold,
        "encoder_sha256": sha256(encoder_path),
        "holdout_aggregate":
            holdout_aggregate,
        "limitations": [
            "Holdout uses only two independent runs.",
            "Only one independent attack holdout.",
            "Train and holdout differ by protocol version.",
            "Seed variance is not a confidence interval.",
            "Channel zeroing is a sensitivity diagnostic.",
            "Holdout metrics are not used for model selection.",
            "No robustness conclusion is justified.",
        ],
    }

    (OUT / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("\n==============================")
    print("ML-5 TRAINING COMPLETED")
    print("Models:", len(checkpoint_manifest))
    print("Results:", OUT)
    print("Checkpoints:", CKPT)
    print("==============================")


if __name__ == "__main__":
    main()
