import json
from pathlib import Path

import numpy as np
import torch

from ml3_preprocessing import TrainOnlyScaler


VARIANTS = ("plain", "mask", "mask_age")

CHANNEL_NAMES = [
    "nInInterests_rate",
    "nOutInterests_rate",
    "nInData_rate",
    "nOutData_rate",
    "nInNacks_rate",
    "nOutNacks_rate",
    "nSatisfiedInterests_rate",
    "nUnsatisfiedInterests_rate",
    "nPitEntries",
    "nCsEntries",
    "mask",
    "age_log1p_standardized",
    "age_valid",
]


class MaskAgeEncoder:
    """
    ML-4 causal observation encoder.

    Input:
        Batch from ML-2 temporal dataset.

    Output:
        Tensor [B, T, N, 13].

    Feature scaling comes from ML-3 train-only scaler.
    Age scaling is fitted on training-run timesteps only.

    No late reports or source-audit telemetry are read.
    """

    def __init__(
        self,
        feature_scaler,
        age_mean,
        age_std,
        train_runs,
    ):
        self.feature_scaler = feature_scaler
        self.age_mean = float(age_mean)
        self.age_std = float(age_std)
        self.train_runs = list(train_runs)

        if self.age_std <= 0:
            raise ValueError("Invalid Age std")

    @classmethod
    def fit(
        cls,
        data_dir,
        train_runs,
        feature_scaler_path,
    ):
        feature_scaler = TrainOnlyScaler.load(
            feature_scaler_path
        )

        if feature_scaler.train_runs != list(train_runs):
            raise ValueError(
                "Feature scaler train-run provenance mismatch"
            )

        ages = []

        for run_id in train_runs:
            path = Path(data_dir) / f"{run_id}.npz"

            with np.load(
                path,
                allow_pickle=False
            ) as data:
                age = data["age"]
                valid = data["age_valid"]

                if age.shape != (50, 10):
                    raise ValueError(
                        f"Unexpected Age shape: {run_id}"
                    )

                selected = age[valid > 0.5]

                if not np.isfinite(selected).all():
                    raise ValueError(
                        f"Nonfinite Age: {run_id}"
                    )

                if np.any(selected < 0):
                    raise ValueError(
                        f"Negative Age: {run_id}"
                    )

                ages.append(
                    np.log1p(
                        selected.astype(np.float64)
                    )
                )

        values = np.concatenate(ages)

        if values.size == 0:
            raise ValueError("No valid training Age")

        age_mean = float(values.mean())
        age_std = max(float(values.std()), 1e-6)

        return cls(
            feature_scaler=feature_scaler,
            age_mean=age_mean,
            age_std=age_std,
            train_runs=train_runs,
        )

    def encode(self, batch, variant):
        if variant not in VARIANTS:
            raise ValueError(
                f"Unknown variant: {variant}"
            )

        required = [
            "x",
            "mask",
            "age",
            "age_valid",
            "feature_valid",
        ]

        for key in required:
            if key not in batch:
                raise KeyError(key)

        x = batch["x"]
        mask = batch["mask"]
        age = batch["age"]
        age_valid = batch["age_valid"]
        feature_valid = batch["feature_valid"]

        if x.ndim != 4 or x.shape[-2:] != (10, 10):
            raise ValueError("Expected X [B,T,10,10]")

        expected_shape = x.shape[:-1]

        for name, tensor in [
            ("mask", mask),
            ("age", age),
            ("age_valid", age_valid),
        ]:
            if tensor.shape != expected_shape:
                raise ValueError(
                    f"Unexpected {name} shape"
                )

        if feature_valid.shape != x.shape:
            raise ValueError(
                "Unexpected feature_valid shape"
            )

        for name, tensor in [
            ("x", x),
            ("mask", mask),
            ("age", age),
            ("age_valid", age_valid),
            ("feature_valid", feature_valid),
        ]:
            if not torch.isfinite(tensor).all():
                raise ValueError(
                    f"Nonfinite tensor: {name}"
                )

        for name, tensor in [
            ("mask", mask),
            ("age_valid", age_valid),
            ("feature_valid", feature_valid),
        ]:
            if not torch.all(
                (tensor == 0) | (tensor == 1)
            ):
                raise ValueError(
                    f"Invalid binary tensor: {name}"
                )

        if torch.any(age < 0):
            raise ValueError("Negative Age")

        if not torch.all(
            feature_valid <= mask.unsqueeze(-1)
        ):
            raise ValueError(
                "Unavailable report has valid features"
            )

        if not torch.all(
            x[feature_valid == 0] == 0
        ):
            raise ValueError(
                "Invalid feature placeholder must be zero"
            )

        if not torch.all(
            age[age_valid == 0] == 0
        ):
            raise ValueError(
                "Invalid Age placeholder must be zero"
            )

        # Identical 10-feature representation for all arms.
        telemetry = self.feature_scaler.transform(
            x, feature_valid
        )

        zeros = torch.zeros_like(mask)

        if variant == "plain":
            extra = torch.stack(
                [zeros, zeros, zeros],
                dim=-1
            )

        elif variant == "mask":
            extra = torch.stack(
                [mask, zeros, zeros],
                dim=-1
            )

        else:
            age_normalized = (
                torch.log1p(age) - self.age_mean
            ) / self.age_std

            age_normalized = torch.where(
                age_valid > 0.5,
                age_normalized,
                zeros,
            )

            extra = torch.stack(
                [mask, age_normalized, age_valid],
                dim=-1
            )

        encoded = torch.cat(
            [telemetry, extra],
            dim=-1
        )

        if encoded.shape[-1] != 13:
            raise RuntimeError(
                "Expected 13 input channels"
            )

        if not torch.isfinite(encoded).all():
            raise ValueError(
                "Nonfinite encoded features"
            )

        return encoded

    def save(self, path):
        metadata = {
            "method": "ML4_causal_mask_age",
            "input_dim": 13,
            "channel_names": CHANNEL_NAMES,
            "train_runs": self.train_runs,
            "age_transform": "log1p_standardization",
            "age_mean": self.age_mean,
            "age_std": self.age_std,
            "feature_mean": self.feature_scaler.mean,
            "feature_std": self.feature_scaler.std,
            "missing_policy": (
                "Missing feature values zero-filled "
                "after feature scaling; invalid Age zero-filled "
                "after Age scaling, with age_valid retained."
            ),
        }

        Path(path).write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        return metadata

    @classmethod
    def load(cls, path):
        obj = json.loads(
            Path(path).read_text(encoding="utf-8")
        )

        scaler = TrainOnlyScaler()
        scaler.mean = obj["feature_mean"]
        scaler.std = obj["feature_std"]
        scaler.feature_count = len(obj["feature_mean"])
        scaler.train_runs = obj["train_runs"]

        return cls(
            feature_scaler=scaler,
            age_mean=obj["age_mean"],
            age_std=obj["age_std"],
            train_runs=obj["train_runs"],
        )
