import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from observed_temporal_dataset import (
    ObservedTemporalDataset
)

from train_graphsage_gru import (
    GraphSAGEGRU,
    collate_temporal,
    run_epoch,
)


SEED = 42
PERTURBATION_SEED = 2026
LOSS_RATE = 0.20

EPOCHS = 30
BATCH_SIZE = 16
HIDDEN_DIM = 64

OUT_ROOT = Path(
    "results/ablation_random_loss20_seed2026"
)

INPUT_DIMS = {
    "plain": 12,
    "mask": 13,
    "mask_age": 14,
}


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(split, mode):
    return ObservedTemporalDataset(
        split=split,
        mode=mode,
        loss_rate=LOSS_RATE,
        delay_rate=0.0,
        seed=PERTURBATION_SEED,
    )


def make_loader(dataset, shuffle=False):
    generator = torch.Generator()
    generator.manual_seed(SEED)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=collate_temporal,
        generator=generator,
    )


def make_criterion(train_dataset, device):
    labels = torch.tensor([
        window[2]
        for window in train_dataset.windows
    ])

    counts = torch.bincount(
        labels,
        minlength=2
    ).float()

    weights = counts.sum() / (
        2 * counts.clamp(min=1)
    )

    return nn.CrossEntropyLoss(
        weight=weights.to(device)
    )


def train_variant(mode, device):
    print(f"\n{'=' * 55}")
    print(f"TRAINING VARIANT: {mode.upper()}")
    print(f"{'=' * 55}")

    set_seed(SEED)

    train_ds = make_dataset(
        "train",
        mode
    )

    val_ds = make_dataset(
        "val",
        mode
    )

    test_ds = make_dataset(
        "test",
        mode
    )

    train_loader = make_loader(
        train_ds,
        shuffle=True
    )

    val_loader = make_loader(val_ds)
    test_loader = make_loader(test_ds)

    model = GraphSAGEGRU(
        input_dim=INPUT_DIMS[mode],
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    criterion = make_criterion(
        train_ds,
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=5e-4,
    )

    out = OUT_ROOT / mode

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_path = (
        out / "best_model.pt"
    )

    best_val_loss = float("inf")
    best_epoch = None
    history = []

    for epoch in range(
        1,
        EPOCHS + 1
    ):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
        })

        if (
            val_metrics["loss"]
            < best_val_loss
        ):
            best_val_loss = (
                val_metrics["loss"]
            )

            best_epoch = epoch

            torch.save({
                "model_state_dict": (
                    model.state_dict()
                ),
                "epoch": epoch,
                "mode": mode,
                "input_dim": (
                    INPUT_DIMS[mode]
                ),
                "window": 5,
                "seed": SEED,
                "perturbation_seed": (
                    PERTURBATION_SEED
                ),
                "loss_rate": LOSS_RATE,
            }, checkpoint_path)

        print(
            f"{mode:8s} | "
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train F1="
            f"{train_metrics['f1_ifa']:.4f} | "
            f"Val F1="
            f"{val_metrics['f1_ifa']:.4f} | "
            f"Val loss="
            f"{val_metrics['loss']:.6f}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    val_metrics = run_epoch(
        model,
        val_loader,
        criterion,
        device
    )

    test_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        device
    )

    result = {
        "model": "GraphSAGE-GRU",
        "mode": mode,
        "dataset": "SPOTIFAI",
        "scenario": (
            "non_existing_small_pilot"
        ),
        "window": 5,
        "loss_type": (
            "independent_random"
        ),
        "loss_rate": LOSS_RATE,
        "seed": SEED,
        "perturbation_seed": (
            PERTURBATION_SEED
        ),
        "best_epoch": best_epoch,
        "validation": val_metrics,
        "test": test_metrics,
    }

    with open(
        out / "history.json",
        "w"
    ) as f:
        json.dump(
            history,
            f,
            indent=2
        )

    with open(
        out / "metrics.json",
        "w"
    ) as f:
        json.dump(
            result,
            f,
            indent=2
        )

    print(
        f"\n===== FINAL TEST: "
        f"{mode.upper()} ====="
    )

    print(
        json.dumps(
            test_metrics,
            indent=2
        )
    )

    return result


def evaluate_clean_checkpoint(device):
    print(
        "\n===== CLEAN MODEL UNDER 20% LOSS ====="
    )

    checkpoint_path = Path(
        "results/graphsage_gru_clean_pilot/"
        "best_model.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Clean checkpoint missing: "
            f"{checkpoint_path}"
        )

    set_seed(SEED)

    train_ds = make_dataset(
        "train",
        "plain"
    )

    test_ds = make_dataset(
        "test",
        "plain"
    )

    test_loader = make_loader(
        test_ds
    )

    criterion = make_criterion(
        train_ds,
        device
    )

    model = GraphSAGEGRU(
        input_dim=12,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    metrics = run_epoch(
        model,
        test_loader,
        criterion,
        device
    )

    result = {
        "model": "GraphSAGE-GRU",
        "mode": (
            "clean_pretrained_on_loss20"
        ),
        "training_condition": "clean",
        "testing_condition": (
            "20pct_random_report_loss"
        ),
        "test": metrics,
    }

    print(
        json.dumps(
            metrics,
            indent=2
        )
    )

    return result


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)
    print("Loss rate:", LOSS_RATE)
    print(
        "Perturbation seed:",
        PERTURBATION_SEED
    )

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    val_check = make_dataset(
        "val",
        "plain"
    )

    test_check = make_dataset(
        "test",
        "plain"
    )

    assert not torch.equal(
        val_check.observations[0]["mask"],
        test_check.observations[0]["mask"],
    ), "Validation/test perturbations identical"

    print(
        "Validation/test perturbations distinct: PASSED"
    )

    results = {}

    results["clean_pretrained"] = (
        evaluate_clean_checkpoint(device)
    )

    for mode in (
        "plain",
        "mask",
        "mask_age"
    ):
        results[mode] = (
            train_variant(
                mode,
                device
            )
        )

    with open(
        OUT_ROOT / "summary.json",
        "w"
    ) as f:
        json.dump(
            results,
            f,
            indent=2
        )

    print("\n" + "=" * 75)
    print("ABLATION SUMMARY - TEST SET")
    print("=" * 75)

    for name, result in results.items():
        metrics = result["test"]

        print(
            f"{name:20s} | "
            f"Acc={metrics['accuracy']:.4f} | "
            f"F1={metrics['f1_ifa']:.4f} | "
            f"Recall={metrics['recall_ifa']:.4f} | "
            f"FPR="
            f"{metrics['false_positive_rate']:.4f}"
        )

    print("\nResults saved:", OUT_ROOT)


if __name__ == "__main__":
    main()