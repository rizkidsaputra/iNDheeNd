from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data.separate import separate

from monitoring_observation import simulate_monitoring


ROOT = Path(
    "data/baseline_cache/non_existing/small/processed"
)

RUN_IDS = {
    "train": [1, 2, 3],
    "val": [4],
    "test": [5],
}

WINDOW = 5
MAX_AGE = 10


def normalize_observed(x):
    x = x.float()
    x = x - x.min()

    return x / x.sum(
        dim=-1,
        keepdim=True
    ).clamp(min=1.0)


class ObservedTemporalDataset(Dataset):

    def __init__(
        self,
        split,
        mode="plain",
        loss_rate=0.2,
        delay_rate=0.0,
        delay_seconds=2,
        seed=2026,
    ):
        assert split in RUN_IDS
        assert mode in (
            "plain", "mask", "mask_age"
        )

        assert 0 <= loss_rate <= 1
        assert 0 <= delay_rate <= 1
        assert delay_seconds >= 0

        paths = list(
            ROOT.glob(f"{split}_*_data.pt")
        )

        assert len(paths) == 1, paths

        obj = torch.load(
            paths[0],
            map_location="cpu",
            weights_only=False,
        )

        data, slices = obj[:2]

        count = len(slices["x"]) - 1

        graphs = [
            separate(
                cls=data.__class__,
                batch=data,
                idx=i,
                slice_dict=slices,
                decrement=False,
            )
            for i in range(count)
        ]

        times = [
            int(g.time) for g in graphs
        ]

        boundaries = [0]

        boundaries.extend(
            i for i in range(1, count)
            if times[i] <= times[i - 1]
        )

        boundaries.append(count)

        assert (
            len(boundaries) - 1
            == len(RUN_IDS[split])
        )

        self.runs = []
        self.observations = []
        self.windows = []
        self.mode = mode

        for run_index, (start, end) in enumerate(
            zip(
                boundaries[:-1],
                boundaries[1:]
            )
        ):
            run_id = RUN_IDS[split][run_index]

            run_graphs = graphs[start:end]
            run_times = times[start:end]

            assert run_times == list(
                range(
                    run_times[0],
                    run_times[0] + len(run_times)
                )
            )

            raw_x = torch.stack([
                g.x.float()
                for g in run_graphs
            ])

            T, N, F = raw_x.shape

            generator = torch.Generator()
            generator.manual_seed(
                seed + run_id
            )

            dropped = (
                torch.rand(
                    (T, N),
                    generator=generator
                ) < loss_rate
            )

            delayed = (
                torch.rand(
                    (T, N),
                    generator=generator
                ) < delay_rate
            )

            delays = (
                delayed.long()
                * delay_seconds
            )

            obs = simulate_monitoring(
                raw_x,
                run_times,
                dropped=dropped,
                delays=delays,
            )

            self.observations.append(obs)

            observed_graphs = []

            for t, original_graph in enumerate(
                run_graphs
            ):
                base_x = normalize_observed(
                    obs["observed_x"][t]
                )

                features = [base_x]

                if mode in (
                    "mask",
                    "mask_age"
                ):
                    features.append(
                        obs["mask"][t].unsqueeze(-1)
                    )

                if mode == "mask_age":
                    valid = (
                        obs["has_report"][t].bool()
                    )

                    age_feature = torch.where(
                        valid,
                        obs["age"][t]
                        .clamp(max=MAX_AGE)
                        / (MAX_AGE + 1),
                        torch.ones_like(
                            obs["age"][t]
                        ),
                    )

                    features.append(
                        age_feature.unsqueeze(-1)
                    )

                node_features = torch.cat(
                    features,
                    dim=-1
                )

                assert (
                    node_features.shape[0] == N
                )

                assert torch.isfinite(
                    node_features
                ).all()

                observed_graphs.append(
                    Data(
                        x=node_features,
                        edge_index=(
                            original_graph
                            .edge_index
                            .long()
                        ),
                    )
                )

            self.runs.append(
                observed_graphs
            )

            # Create sliding temporal windows.
            for last in range(
                WINDOW - 1,
                T
            ):
                label = int(
                    run_graphs[last].attack_is_on
                )

                self.windows.append((
                    run_index,
                    last,
                    label
                ))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        run_index, last, label = (
            self.windows[idx]
        )

        sequence = self.runs[run_index][
            last - WINDOW + 1:last + 1
        ]

        return sequence, label