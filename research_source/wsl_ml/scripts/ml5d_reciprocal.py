#!/usr/bin/env python3

import csv
import hashlib
import json
import random
import sys
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
from ml3_preprocessing import TrainOnlyScaler
from ml4_observation import MaskAgeEncoder
from models.graphsage_gru import GraphSAGEGRU

CONFIG = ROOT / "configs/ml5d/reciprocal.json"
DATA = ROOT / "data/processed/ml2"
OUT = ROOT / "results/ml5d"
CKPT = ROOT / "checkpoints/ml5d"


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def save_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, indent=2),
        encoding="utf-8",
    )


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


def build_model(config):
    return GraphSAGEGRU(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        gru_hidden_dim=config["gru_hidden_dim"],
        dropout=config["dropout"],
    )


def model_signature(model):
    # Confirm identical initialization per seed
    # across the three input variants.
    values = torch.cat([
        p.detach().cpu().reshape(-1)
        for p in model.parameters()
    ])
    return hashlib.sha256(
        values.numpy().tobytes()
    ).hexdigest()


def move_batch(batch, device):
    return {
        k: v.to(device)
        if isinstance(v, torch.Tensor)
        else v
        for k, v in batch.items()
    }


def get_labels(run_ids):
    values = []
    for run in run_ids:
        with np.load(
            DATA / f"{run}.npz",
            allow_pickle=False
        ) as d:
            values.append(d["y_window"].copy())
    return np.concatenate(values)


def training_nominal_age(run_ids):
    values = []
    for run in run_ids:
        with np.load(
            DATA / f"{run}.npz",
            allow_pickle=False
        ) as d:
            age = d["age"]
            mask = d["mask"]
            valid = d["age_valid"]

            selected = age[
                (mask == 1) & (valid == 1)
            ]

            assert np.isfinite(selected).all()
            assert np.all(selected >= 0)
            values.append(selected)

    result = float(
        np.median(np.concatenate(values))
    )

    return result


def metrics(rows, threshold):
    y = np.array(
        [r["label"] for r in rows],
        dtype=np.int64
    )

    p = np.array(
        [r["probability"] for r in rows],
        dtype=np.float64
    )

    assert len(y) > 0
    assert np.isfinite(p).all()
    assert np.all((p >= 0) & (p <= 1))

    pred = (p >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    positive = bool(np.any(y == 1))
    negative = bool(np.any(y == 0))

    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "negatives": int((y == 0).sum()),
        "accuracy": float(
            accuracy_score(y, pred)
        ),
        "precision": (
            float(precision_score(
                y, pred, zero_division=0
            ))
            if positive else None
        ),
        "recall": (
            float(recall_score(
                y, pred, zero_division=0
            ))
            if positive else None
        ),
        "f1": (
            float(f1_score(
                y, pred, zero_division=0
            ))
            if positive else None
        ),
        "fpr": (
            float(fp / (fp + tn))
            if negative else None
        ),
        "ap": (
            float(average_precision_score(y, p))
            if positive and negative else None
        ),
        "roc_auc": (
            float(roc_auc_score(y, p))
            if positive and negative else None
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    encoder,
    variant,
    device,
    edges,
    threshold,
    condition,
    nominal_age,
):
    model.eval()
    rows = []

    for batch in loader:
        b = move_batch(batch, device)

        if condition == "nominal_missing":
            modified = dict(b)
            age = b["age"].clone()

            replace = (
                (b["mask"] == 0)
                & (b["age_valid"] == 1)
            )

            age = torch.where(
                replace,
                torch.full_like(age, nominal_age),
                age,
            )

            modified["age"] = age
            b = modified

        elif condition != "original":
            raise ValueError(condition)

        x = encoder.encode(b, variant)

        probs = torch.sigmoid(
            model(x, edges)
        ).cpu().numpy()

        labels = batch["y"].cpu().numpy()
        sequences = batch[
            "window_end_sequence"
        ].cpu().numpy()

        for i in range(len(probs)):
            rows.append({
                "run_id": batch["run_id"][i],
                "end_sequence": int(sequences[i]),
                "label": int(labels[i]),
                "probability": float(probs[i]),
                "prediction": int(
                    probs[i] >= threshold
                ),
                "missing_at_decision": int(
                    (batch["mask"][i, -1] == 0)
                    .sum().item()
                ),
            })

    return rows


def main():
    config = read_json(CONFIG)

    assert config["variants"] == [
        "plain", "mask", "mask_age"
    ]
    assert config["checkpoint_selection"] == "final_epoch"
    assert config["selection_using_holdout"] is False

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)

    all_predictions = []
    all_metrics = []
    histories = []
    manifest = []
    directions_info = []
    init_signatures = {}

    for direction, split in config["directions"].items():
        train_runs = split["train_runs"]
        holdout_runs = split["holdout_runs"]

        assert set(train_runs).isdisjoint(
            holdout_runs
        )
        assert len(train_runs) == 2
        assert len(holdout_runs) == 2

        train_ds = NDNTemporalGraphDataset(
            data_dir=DATA,
            run_ids=train_runs,
        )

        holdout_ds = NDNTemporalGraphDataset(
            data_dir=DATA,
            run_ids=holdout_runs,
        )

        assert len(train_ds) == 92
        assert len(holdout_ds) == 92
        assert torch.equal(
            train_ds.edge_index,
            holdout_ds.edge_index
        )

        train_y = get_labels(train_runs)
        holdout_y = get_labels(holdout_runs)

        assert len(train_y) == 92
        assert len(holdout_y) == 92
        assert int(train_y.sum()) == 15
        assert int(holdout_y.sum()) == 15

        pos_weight = float(
            (len(train_y) - train_y.sum())
            / train_y.sum()
        )

        direction_out = OUT / direction
        direction_ckpt = CKPT / direction

        direction_out.mkdir(
            parents=True, exist_ok=True
        )
        direction_ckpt.mkdir(
            parents=True, exist_ok=True
        )

        # Both feature and Age scalers are fitted
        # independently using ONLY this direction's
        # training runs.
        feature_path = (
            direction_out / "feature_scaler.json"
        )
        encoder_path = (
            direction_out / "encoder.json"
        )

        scaler = TrainOnlyScaler().fit(
            DATA, train_runs
        )
        scaler.save(feature_path)

        encoder = MaskAgeEncoder.fit(
            data_dir=DATA,
            train_runs=train_runs,
            feature_scaler_path=feature_path,
        )
        encoder.save(encoder_path)

        nominal_age = training_nominal_age(
            train_runs
        )

        edges = train_ds.edge_index.to(device)

        train_eval_loader = DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=0,
        )

        holdout_loader = DataLoader(
            holdout_ds,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=0,
        )

        directions_info.append({
            "direction": direction,
            "train_runs": train_runs,
            "holdout_runs": holdout_runs,
            "training_windows": len(train_ds),
            "holdout_windows": len(holdout_ds),
            "training_positives": int(
                train_y.sum()
            ),
            "holdout_positives": int(
                holdout_y.sum()
            ),
            "pos_weight": pos_weight,
            "training_nominal_age_s": nominal_age,
            "feature_scaler_sha256": sha256(
                feature_path
            ),
            "encoder_sha256": sha256(
                encoder_path
            ),
        })

        print("\n==================================")
        print("DIRECTION:", direction)
        print("Train:", train_runs)
        print("Holdout:", holdout_runs)
        print("Device:", device)
        print("Windows: 92 train / 92 holdout")
        print("Pos weight:", pos_weight)
        print("Nominal Age:", nominal_age)
        print("==================================")

        for seed in config["seeds"]:
            for variant in config["variants"]:
                # Reset RNG before every variant.
                set_seed(seed)

                model = build_model(config).to(
                    device
                )

                n_parameters = sum(
                    p.numel()
                    for p in model.parameters()
                )

                assert n_parameters == 35009

                signature = model_signature(model)

                key = (direction, seed)

                if key not in init_signatures:
                    init_signatures[key] = signature
                else:
                    assert (
                        init_signatures[key] == signature
                    ), "Different initialization"

                generator = torch.Generator()
                generator.manual_seed(seed)

                train_loader = DataLoader(
                    train_ds,
                    batch_size=config["batch_size"],
                    shuffle=True,
                    generator=generator,
                    num_workers=0,
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
                    weight_decay=config[
                        "weight_decay"
                    ],
                )

                print(
                    f"\n[{direction}] "
                    f"seed={seed}, variant={variant}"
                )

                for epoch in range(
                    1, config["epochs"] + 1
                ):
                    model.train()

                    total_loss = 0.0
                    total_count = 0

                    for batch in train_loader:
                        b = move_batch(
                            batch, device
                        )

                        x = encoder.encode(
                            b, variant
                        )
                        y = b["y"].float()

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
                            max_norm=1.0,
                        )

                        optimizer.step()

                        size = len(y)
                        total_count += size
                        total_loss += (
                            loss.item() * size
                        )

                    mean_loss = (
                        total_loss / total_count
                    )

                    histories.append({
                        "direction": direction,
                        "seed": seed,
                        "variant": variant,
                        "epoch": epoch,
                        "train_loss": mean_loss,
                    })

                    print(
                        f"Epoch {epoch:02d}: "
                        f"loss={mean_loss:.6f}"
                    )

                checkpoint_path = (
                    direction_ckpt
                    / f"{variant}_seed{seed}.pt"
                )

                torch.save({
                    "model_state_dict":
                        model.state_dict(),
                    "model_config": {
                        "input_dim": 13,
                        "hidden_dim": config[
                            "hidden_dim"
                        ],
                        "gru_hidden_dim": config[
                            "gru_hidden_dim"
                        ],
                        "dropout": config["dropout"],
                    },
                    "direction": direction,
                    "variant": variant,
                    "seed": seed,
                    "epoch": config["epochs"],
                    "train_runs": train_runs,
                    "holdout_runs": holdout_runs,
                    "encoder_sha256": sha256(
                        encoder_path
                    ),
                    "initialization_sha256":
                        signature,
                    "nominal_age_s": nominal_age,
                }, checkpoint_path)

                manifest.append({
                    "direction": direction,
                    "seed": seed,
                    "variant": variant,
                    "checkpoint": str(
                        checkpoint_path.relative_to(
                            ROOT
                        )
                    ),
                    "sha256": sha256(
                        checkpoint_path
                    ),
                    "initialization_sha256":
                        signature,
                    "encoder_sha256": sha256(
                        encoder_path
                    ),
                    "parameters": n_parameters,
                })

                for split_name, loader in [
                    ("train", train_eval_loader),
                    ("holdout", holdout_loader),
                ]:
                    conditions = (
                        ["original"]
                        if split_name == "train"
                        else [
                            "original",
                            "nominal_missing",
                        ]
                    )

                    run_list = (
                        train_runs
                        if split_name == "train"
                        else holdout_runs
                    )

                    for condition in conditions:
                        rows = evaluate(
                            model,
                            loader,
                            encoder,
                            variant,
                            device,
                            edges,
                            config["threshold"],
                            condition,
                            nominal_age,
                        )

                        for row in rows:
                            row.update({
                                "direction": direction,
                                "seed": seed,
                                "variant": variant,
                                "split": split_name,
                                "condition": condition,
                            })

                        all_predictions.extend(rows)

                        for run_id in [
                            "ALL", *run_list
                        ]:
                            selected = (
                                rows
                                if run_id == "ALL"
                                else [
                                    r for r in rows
                                    if r["run_id"] == run_id
                                ]
                            )

                            all_metrics.append({
                                "direction": direction,
                                "seed": seed,
                                "variant": variant,
                                "split": split_name,
                                "condition": condition,
                                "run_id": run_id,
                                **metrics(
                                    selected,
                                    config["threshold"]
                                ),
                            })

                original = [
                    r for r in all_metrics
                    if r["direction"] == direction
                    and r["seed"] == seed
                    and r["variant"] == variant
                    and r["split"] == "holdout"
                    and r["condition"] == "original"
                    and r["run_id"] == "ALL"
                ]

                assert len(original) == 1

                print(
                    "Holdout F1:",
                    round(original[0]["f1"], 4),
                    "Recall:",
                    round(original[0]["recall"], 4),
                )

    assert len(manifest) == 18
    assert len(histories) == 18 * 20

    prediction_cols = [
        "direction", "seed", "variant",
        "split", "condition", "run_id",
        "end_sequence", "label", "probability",
        "prediction", "missing_at_decision",
    ]

    metric_cols = [
        "direction", "seed", "variant",
        "split", "condition", "run_id",
        "n", "positives", "negatives",
        "accuracy", "precision", "recall",
        "f1", "fpr", "ap", "roc_auc",
        "tn", "fp", "fn", "tp",
    ]

    write_csv(
        OUT / "all_predictions.csv",
        all_predictions,
        prediction_cols
    )

    write_csv(
        OUT / "metrics_by_run.csv",
        all_metrics,
        metric_cols
    )

    write_csv(
        OUT / "training_history.csv",
        histories,
        [
            "direction", "seed", "variant",
            "epoch", "train_loss",
        ]
    )

    save_json(
        OUT / "checkpoints.json",
        manifest
    )

    save_json(
        OUT / "split_and_scaler_audit.json",
        directions_info
    )

    save_json(
        OUT / "experiment_config.json",
        config
    )

    print("\n============================")
    print("ML-5D TRAINING COMPLETED")
    print("Directions: 2")
    print("Models trained: 18")
    print("Checkpoints:", CKPT)
    print("Results:", OUT)
    print("============================")


if __name__ == "__main__":
    main()
