from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class NDNTemporalGraphDataset(Dataset):
    """
    Per-run temporal graph dataset.

    No automatic run splitting.
    No scaling or model-specific imputation.
    """

    def __init__(
        self,
        data_dir="data/processed/ml2",
        run_ids=None
    ):
        if not run_ids:
            raise ValueError(
                "Explicit run_ids required to avoid "
                "accidental run mixing."
            )

        self.samples = []
        self.edge_index = None

        data_dir = Path(data_dir)

        for run_id in run_ids:
            path = data_dir / f"{run_id}.npz"

            if not path.is_file():
                raise FileNotFoundError(path)

            with np.load(path, allow_pickle=False) as d:
                edges = d["edge_index"].copy()

                if self.edge_index is None:
                    self.edge_index = edges
                elif not np.array_equal(
                    self.edge_index, edges
                ):
                    raise ValueError(
                        "Inconsistent graph topology"
                    )

                x = d["X_window"].copy()
                mask = d["mask_window"].copy()
                age = d["age_window"].copy()
                age_valid = d["age_valid_window"].copy()
                feature_valid = (
                    d["feature_valid_window"].copy()
                )
                y = d["y_window"].copy()

            if x.ndim != 4 or x.shape[1:] != (5, 10, 10):
                raise ValueError(
                    f"Invalid tensor shape in {run_id}"
                )

            if not (
                len(x) == len(mask)
                == len(age) == len(y)
            ):
                raise ValueError(
                    f"Inconsistent window count: {run_id}"
                )

            for i in range(len(y)):
                self.samples.append({
                    "x": x[i],
                    "mask": mask[i],
                    "age": age[i],
                    "age_valid": age_valid[i],
                    "feature_valid": feature_valid[i],
                    "y": int(y[i]),
                    "run_id": run_id,
                    "window_end_sequence": i + 5,
                })

        self.edge_index = torch.as_tensor(
            self.edge_index,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        return {
            "x": torch.tensor(
                s["x"], dtype=torch.float32
            ),
            "mask": torch.tensor(
                s["mask"], dtype=torch.float32
            ),
            "age": torch.tensor(
                s["age"], dtype=torch.float32
            ),
            "age_valid": torch.tensor(
                s["age_valid"], dtype=torch.float32
            ),
            "feature_valid": torch.tensor(
                s["feature_valid"], dtype=torch.float32
            ),
            "y": torch.tensor(
                s["y"], dtype=torch.long
            ),
            "edge_index": self.edge_index,
            "run_id": s["run_id"],
            "window_end_sequence": (
                s["window_end_sequence"]
            ),
        }
