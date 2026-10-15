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
from ml4_observation import MaskAgeEncoder, VARIANTS
from models.graphsage_gru import GraphSAGEGRU


CONFIG = ROOT / "configs/ml3/smoke.json"
DATA = ROOT / "data/processed/ml2"
OUT = ROOT / "results/ml4"

OUT.mkdir(parents=True, exist_ok=True)


def main():
    config = json.loads(
        CONFIG.read_text(encoding="utf-8")
    )

    train_runs = config["train_runs"]
    holdout_runs = config[
        "diagnostic_holdout_runs"
    ]

    assert set(train_runs).isdisjoint(
        set(holdout_runs)
    )

    # Fit exclusively on training runs.
    encoder = MaskAgeEncoder.fit(
        data_dir=DATA,
        train_runs=train_runs,
        feature_scaler_path=(
            ROOT / "results/ml3/scaler.json"
        ),
    )

    metadata_path = OUT / "encoder.json"
    encoder.save(metadata_path)

    restored = MaskAgeEncoder.load(
        metadata_path
    )

    assert restored.train_runs == train_runs
    assert encoder.age_mean == restored.age_mean
    assert encoder.age_std == restored.age_std

    np.testing.assert_allclose(
        encoder.feature_scaler.mean,
        restored.feature_scaler.mean,
    )

    np.testing.assert_allclose(
        encoder.feature_scaler.std,
        restored.feature_scaler.std,
    )

    print("[PASS] Train-only encoder provenance")
    print("[PASS] Encoder serialization")

    train_dataset = NDNTemporalGraphDataset(
        data_dir=DATA,
        run_ids=train_runs,
    )

    holdout_dataset = NDNTemporalGraphDataset(
        data_dir=DATA,
        run_ids=holdout_runs,
    )

    assert len(train_dataset) == 184
    assert len(holdout_dataset) == 92

    print("[PASS] Run-level split")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    edge_index = train_dataset.edge_index.to(
        device
    )

    summaries = {}
    test_batches = []

    # Read-only evaluation of tensor validity.
    for split_name, dataset in [
        ("train", train_dataset),
        ("holdout", holdout_dataset),
    ]:
        loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
        )

        for batch in loader:
            test_batches.append((split_name, batch))

    for split_name, batch in test_batches:
        gpu_batch = {
            key: value.to(device)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in batch.items()
        }

        encoded = {
            variant: restored.encode(
                gpu_batch, variant
            )
            for variant in VARIANTS
        }

        original = encoded["plain"]
        with_mask = encoded["mask"]
        with_age = encoded["mask_age"]

        B, T, N, F = original.shape

        assert (T, N, F) == (5, 10, 13)

        # The 10 telemetry channels must be identical.
        torch.testing.assert_close(
            original[..., :10],
            with_mask[..., :10]
        )

        torch.testing.assert_close(
            original[..., :10],
            with_age[..., :10]
        )

        # Plain receives no explicit Mask/Age.
        assert torch.all(
            original[..., 10:] == 0
        )

        # Mask gets current availability only.
        torch.testing.assert_close(
            with_mask[..., 10],
            gpu_batch["mask"]
        )

        assert torch.all(
            with_mask[..., 11:] == 0
        )

        # Mask+Age gets availability and valid ages.
        torch.testing.assert_close(
            with_age[..., 10],
            gpu_batch["mask"]
        )

        torch.testing.assert_close(
            with_age[..., 12],
            gpu_batch["age_valid"]
        )

        assert torch.all(
            with_age[..., 11][
                gpu_batch["age_valid"] == 0
            ] == 0
        )

        # Invalid telemetry stays zero after scaling.
        assert torch.all(
            original[..., :10][
                gpu_batch["feature_valid"] == 0
            ] == 0
        )

        # No future information introduced through
        # altered Age values: Plain/Mask must not change.
        modified = dict(gpu_batch)
        modified_age = gpu_batch["age"].clone()

        modified_age = (
            modified_age
            + 10.0 * gpu_batch["age_valid"]
        )

        modified["age"] = modified_age

        plain_modified = restored.encode(
            modified, "plain"
        )

        mask_modified = restored.encode(
            modified, "mask"
        )

        torch.testing.assert_close(
            original, plain_modified
        )

        torch.testing.assert_close(
            with_mask, mask_modified
        )

        assert torch.isfinite(with_age).all()

    print("[PASS] Three variants share telemetry")
    print("[PASS] Explicit channels isolated")
    print("[PASS] Invalid values remain excluded")
    print("[PASS] Age input isolation")

    # Same model architecture and parameter count
    # across all three variants.
    parameter_counts = {}

    for variant in VARIANTS:
        torch.manual_seed(42)

        if device.type == "cuda":
            torch.cuda.manual_seed_all(42)

        model = GraphSAGEGRU(
            input_dim=13,
            hidden_dim=config["hidden_dim"],
            gru_hidden_dim=config["gru_hidden_dim"],
            dropout=config["dropout"],
        ).to(device)

        model.train()

        sample_batch = next(iter(DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=False,
        )))

        sample_batch = {
            key: value.to(device)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in sample_batch.items()
        }

        x = restored.encode(
            sample_batch,
            variant
        )

        logits = model(x, edge_index)

        assert logits.shape == (8,)
        assert torch.isfinite(logits).all()

        y = sample_batch["y"].float()

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y
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

        count = sum(
            p.numel()
            for p in model.parameters()
        )

        parameter_counts[variant] = count

        print(
            f"[PASS] {variant}: "
            f"input={tuple(x.shape)}, "
            f"output={tuple(logits.shape)}, "
            f"parameters={count}"
        )

    assert len(
        set(parameter_counts.values())
    ) == 1

    print("[PASS] Identical parameter counts")

    # Confirm that Age is meaningful even when a
    # current monitoring report is unavailable.
    age_available_on_missing = 0

    for path in sorted(DATA.glob("*.npz")):
        with np.load(
            path,
            allow_pickle=False
        ) as d:
            age_available_on_missing += int(
                (
                    (d["mask"] == 0)
                    & (d["age_valid"] == 1)
                ).sum()
            )

    report = {
        "status": "PASSED",
        "variants": list(VARIANTS),
        "input_dim": 13,
        "train_windows": len(train_dataset),
        "diagnostic_holdout_windows": len(
            holdout_dataset
        ),
        "device": str(device),
        "parameter_counts": parameter_counts,
        "age_mean": encoder.age_mean,
        "age_std": encoder.age_std,
        "age_valid_on_missing_slots":
            age_available_on_missing,
        "limitations": [
            "Plain may still infer missingness from placeholders.",
            "This test is not a robustness evaluation.",
            "Age impact requires controlled ML-5 training.",
            "Train and holdout differ in protocol strata.",
        ],
    }

    (OUT / "ml4_validation.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n============================")
    print("ML-4 VALIDATION: PASSED")
    print("Device:", device)
    print("Input channels:", 13)
    print("Variants:", ", ".join(VARIANTS))
    print(
        "Age valid despite Mask=0:",
        age_available_on_missing
    )
    print("============================")


if __name__ == "__main__":
    main()
