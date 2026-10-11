import csv
import json

from pathlib import Path
from statistics import mean

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


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

OUT = Path(
    "results/random_loss_stress_pilot"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

RATES = [
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0,
]

SEEDS = [
    3031,
    3032,
    3033,
]

CHECKPOINTS = {
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


def load_models():
    models = {}

    for name, (
        path,
        input_dim,
        _
    ) in CHECKPOINTS.items():

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        model = GraphSAGEGRU(
            input_dim=input_dim,
            hidden_dim=64,
        ).to(DEVICE)

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.eval()

        models[name] = model

    return models


def make_loader(dataset):
    return DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_temporal,
    )


def main():
    print("Device:", DEVICE)

    models = load_models()

    train_ds = ObservedTemporalDataset(
        "train",
        mode="plain",
        loss_rate=0.0
    )

    labels = torch.tensor([
        window[2]
        for window in train_ds.windows
    ])

    counts = torch.bincount(
        labels,
        minlength=2
    ).float()

    weights = counts.sum() / (
        2 * counts.clamp(min=1)
    )

    criterion = nn.CrossEntropyLoss(
        weight=weights.to(DEVICE)
    )

    results = []

    for rate in RATES:
        for seed in SEEDS:

            datasets = {
                mode: ObservedTemporalDataset(
                    split="val",
                    mode=mode,
                    loss_rate=rate,
                    delay_rate=0.0,
                    seed=seed,
                )
                for mode in (
                    "plain",
                    "mask",
                    "mask_age"
                )
            }

            a = datasets[
                "plain"
            ].observations[0]

            for mode in (
                "mask",
                "mask_age"
            ):
                b = datasets[
                    mode
                ].observations[0]

                assert torch.equal(
                    a["observed_x"],
                    b["observed_x"]
                )

                assert torch.equal(
                    a["mask"],
                    b["mask"]
                )

            loaders = {
                mode: make_loader(ds)
                for mode, ds in datasets.items()
            }

            for name, (
                _,
                _,
                mode
            ) in CHECKPOINTS.items():

                metrics = run_epoch(
                    models[name],
                    loaders[mode],
                    criterion,
                    DEVICE,
                )

                results.append({
                    "model": name,
                    "requested_loss_rate": rate,
                    "actual_loss_rate": (
                        1.0
                        - a["mask"].mean().item()
                    ),
                    "seed": seed,
                    **metrics,
                })

                print(
                    f"loss={rate:.0%} | "
                    f"seed={seed} | "
                    f"{name:16s} | "
                    f"F1={metrics['f1_ifa']:.4f} | "
                    f"BA={metrics['balanced_accuracy']:.4f} | "
                    f"FPR="
                    f"{metrics['false_positive_rate']:.4f}",
                    flush=True,
                )

    with open(
        OUT / "detailed.json",
        "w"
    ) as f:
        json.dump(
            results,
            f,
            indent=2
        )

    summary = []

    print(
        "\n===== STRESS TEST SUMMARY (VALIDATION) ====="
    )

    for rate in RATES:
        print(f"\nReport loss: {rate:.0%}")

        for name in CHECKPOINTS:
            subset = [
                r for r in results
                if (
                    r["requested_loss_rate"]
                    == rate
                    and r["model"] == name
                )
            ]

            row = {
                "report_loss": rate,
                "model": name,
                "mean_f1": mean(
                    r["f1_ifa"]
                    for r in subset
                ),
                "mean_balanced_accuracy": mean(
                    r["balanced_accuracy"]
                    for r in subset
                ),
                "mean_recall": mean(
                    r["recall_ifa"]
                    for r in subset
                ),
                "mean_fpr": mean(
                    r["false_positive_rate"]
                    for r in subset
                ),
            }

            summary.append(row)

            print(
                f"{name:16s} | "
                f"F1={row['mean_f1']:.4f} | "
                f"BA={row['mean_balanced_accuracy']:.4f} | "
                f"FPR={row['mean_fpr']:.4f}"
            )

    with open(
        OUT / "summary.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=summary[0].keys()
        )

        writer.writeheader()
        writer.writerows(summary)

    print("\nResults saved:", OUT)


if __name__ == "__main__":
    main()