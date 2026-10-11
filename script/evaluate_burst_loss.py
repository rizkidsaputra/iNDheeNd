import csv
from pathlib import Path

import torch

from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

from monitoring_observation import simulate_monitoring

from observed_temporal_dataset import (
    ObservedTemporalDataset,
    normalize_observed,
    MAX_AGE,
)

from train_graphsage_gru import (
    GraphSAGEGRU,
    collate_temporal,
)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

OUT = Path(
    "results/burst_loss_pilot"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

MODELS = {
    "clean_pretrained": (
        "results/graphsage_gru_clean_pilot/"
        "best_model.pt",
        12,
        "plain"
    ),
    "plain": (
        "results/ablation_random_loss20_seed2026/"
        "plain/best_model.pt",
        12,
        "plain"
    ),
    "mask": (
        "results/ablation_random_loss20_seed2026/"
        "mask/best_model.pt",
        13,
        "mask"
    ),
    "mask_age": (
        "results/ablation_random_loss20_seed2026/"
        "mask_age/best_model.pt",
        14,
        "mask_age"
    ),
}

# Mapping validated for the pilot topology.
ROUT1 = 1
ROUT2 = 4
ROUT5 = 6

SCENARIOS = [
    ("clean", [], 0, 0),

    (
        "normal_all_20s",
        list(range(9)),
        10,
        29
    ),

    (
        "onset_all_20s",
        list(range(9)),
        45,
        64
    ),

    (
        "onset_all_40s",
        list(range(9)),
        45,
        84
    ),

    (
        "onset_rout5_60s",
        [ROUT5],
        45,
        104
    ),

    (
        "onset_three_60s",
        [ROUT1, ROUT2, ROUT5],
        45,
        104
    ),
]


def load_model(path, input_dim):
    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    model = GraphSAGEGRU(
        input_dim=input_dim,
        hidden_dim=64
    ).to(DEVICE)

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    model.eval()

    return model


def make_dataset(mode, dropped):
    ds = ObservedTemporalDataset(
        "val",
        mode=mode,
        loss_rate=0.0
    )

    raw_x = (
        ds.observations[0]["observed_x"]
        .clone()
    )

    times = list(
        range(
            1,
            raw_x.shape[0] + 1
        )
    )

    obs = simulate_monitoring(
        raw_x,
        times,
        dropped=dropped
    )

    ds.observations[0] = obs

    for t, graph in enumerate(
        ds.runs[0]
    ):
        features = [
            normalize_observed(
                obs["observed_x"][t]
            )
        ]

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

            age = torch.where(
                valid,
                obs["age"][t]
                .clamp(max=MAX_AGE)
                / (MAX_AGE + 1),
                torch.ones_like(
                    obs["age"][t]
                )
            )

            features.append(
                age.unsqueeze(-1)
            )

        graph.x = torch.cat(
            features,
            dim=-1
        )

    return ds


@torch.no_grad()
def evaluate(model, ds):
    loader = DataLoader(
        ds,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_temporal
    )

    predictions = []
    labels = []

    for batches, y in loader:
        batches = [
            g.to(DEVICE)
            for g in batches
        ]

        logits = model(batches)

        predictions.extend(
            logits.argmax(
                dim=1
            ).cpu().tolist()
        )

        labels.extend(
            y.tolist()
        )

    times = [
        ds.windows[i][1] + 1
        for i in range(len(ds))
    ]

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    ).ravel()

    first_detection = next(
        (
            t
            for t, p in zip(
                times,
                predictions
            )
            if t >= 51 and p == 1
        ),
        None
    )

    onset_indices = [
        i for i, t in enumerate(times)
        if 51 <= t <= 90
    ]

    onset_recall = sum(
        predictions[i] == 1
        for i in onset_indices
    ) / len(onset_indices)

    return {
        "f1": f1_score(
            labels,
            predictions,
            zero_division=0
        ),
        "balanced_accuracy": (
            balanced_accuracy_score(
                labels,
                predictions
            )
        ),
        "fpr": (
            fp / (fp + tn)
            if (fp + tn) else 0.0
        ),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "onset_recall_51_90": (
            onset_recall
        ),
        "first_detection_time": (
            first_detection
        ),
        "detection_delay": (
            first_detection - 51
            if first_detection is not None
            else None
        ),
    }


def main():
    print("Device:", DEVICE)

    models = {
        name: load_model(
            path,
            dim
        )
        for name, (
            path, dim, _
        ) in MODELS.items()
    }

    results = []

    for (
        scenario,
        routers,
        start,
        end
    ) in SCENARIOS:

        dropped = torch.zeros(
            298,
            9,
            dtype=torch.bool
        )

        if routers:
            dropped[
                start - 1:end,
                routers
            ] = True

        print(
            f"\n===== "
            f"{scenario.upper()} ====="
        )

        datasets = {
            mode: make_dataset(
                mode,
                dropped
            )
            for mode in (
                "plain",
                "mask",
                "mask_age"
            )
        }

        reference = datasets[
            "plain"
        ].observations[0]

        for mode in (
            "mask",
            "mask_age"
        ):
            other = datasets[
                mode
            ].observations[0]

            assert torch.equal(
                reference["observed_x"],
                other["observed_x"]
            )

            assert torch.equal(
                reference["mask"],
                other["mask"]
            )

        for name, (
            _, _, mode
        ) in MODELS.items():

            metrics = evaluate(
                models[name],
                datasets[mode]
            )

            result = {
                "scenario": scenario,
                "model": name,
                **metrics,
            }

            results.append(result)

            print(
                f"{name:17s} "
                f"BA={metrics['balanced_accuracy']:.4f} "
                f"F1={metrics['f1']:.4f} "
                f"FPR={metrics['fpr']:.4f} "
                f"OnsetRecall="
                f"{metrics['onset_recall_51_90']:.3f} "
                f"Delay={metrics['detection_delay']}"
            )

    with open(
        OUT / "results.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        "\nBURST LOSS EVALUATION COMPLETED"
    )

    print(
        "Results:",
        OUT / "results.csv"
    )


if __name__ == "__main__":
    main()