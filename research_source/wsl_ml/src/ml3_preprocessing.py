import json
from pathlib import Path

import numpy as np
import torch


class TrainOnlyScaler:
    """
    log1p + standardization for nonnegative NFD features.

    Fit only on observed training-run timesteps.
    Missing values map to zero AFTER standardization.
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.feature_count = None
        self.train_runs = None

    def fit(self, data_dir, train_runs):
        data_dir = Path(data_dir)
        self.train_runs = list(train_runs)

        if not self.train_runs:
            raise ValueError("No training runs")

        per_feature = None

        for run_id in self.train_runs:
            with np.load(
                data_dir / f"{run_id}.npz",
                allow_pickle=False
            ) as data:
                x = data["X_raw"]
                valid = data["feature_valid"]

                if per_feature is None:
                    per_feature = [
                        [] for _ in range(x.shape[-1])
                    ]

                assert x.shape == valid.shape

                for j in range(x.shape[-1]):
                    values = x[..., j][
                        valid[..., j] > 0.5
                    ]

                    if not np.isfinite(values).all():
                        raise ValueError(
                            f"Nonfinite training values: {run_id}"
                        )

                    if np.any(values < 0):
                        raise ValueError(
                            f"Negative NFD feature: {run_id}"
                        )

                    per_feature[j].append(
                        np.log1p(values.astype(np.float64))
                    )

        means = []
        stds = []

        for j, chunks in enumerate(per_feature):
            values = np.concatenate(chunks)

            if len(values) == 0:
                raise ValueError(
                    f"No valid training values for feature {j}"
                )

            mean = float(np.mean(values))
            std = float(np.std(values))

            means.append(mean)
            stds.append(max(std, 1e-6))

        self.mean = means
        self.std = stds
        self.feature_count = len(means)

        return self

    def transform(self, x, feature_valid):
        if self.mean is None:
            raise RuntimeError("Scaler has not been fitted")

        if x.shape != feature_valid.shape:
            raise ValueError(
                "X and feature_valid shapes differ"
            )

        if x.shape[-1] != self.feature_count:
            raise ValueError("Unexpected feature count")

        if not torch.isfinite(x).all():
            raise ValueError("Nonfinite X")

        if torch.any(x < 0):
            raise ValueError("Negative features")

        mean = torch.as_tensor(
            self.mean,
            dtype=x.dtype,
            device=x.device
        )

        std = torch.as_tensor(
            self.std,
            dtype=x.dtype,
            device=x.device
        )

        transformed = (
            torch.log1p(x) - mean
        ) / std

        transformed = torch.where(
            feature_valid > 0.5,
            transformed,
            torch.zeros_like(transformed)
        )

        if not torch.isfinite(transformed).all():
            raise ValueError("Nonfinite scaled features")

        return transformed

    def save(self, path):
        obj = {
            "method": "log1p_standardization",
            "mean": self.mean,
            "std": self.std,
            "feature_count": self.feature_count,
            "train_runs": self.train_runs,
            "missing_policy": (
                "zero after scaling, no future imputation"
            ),
        }

        Path(path).write_text(
            json.dumps(obj, indent=2)
        )

    @classmethod
    def load(cls, path):
        obj = json.loads(Path(path).read_text())

        scaler = cls()
        scaler.mean = obj["mean"]
        scaler.std = obj["std"]
        scaler.feature_count = obj["feature_count"]
        scaler.train_runs = obj["train_runs"]

        return scaler
