import torch

from observed_temporal_dataset import (
    ObservedTemporalDataset,
    normalize_observed,
    MAX_AGE,
)


MODES = (
    "plain",
    "mask",
    "mask_age"
)


for split in ("train", "val", "test"):
    datasets = {
        mode: ObservedTemporalDataset(
            split=split,
            mode=mode,
            loss_rate=0.2,
            delay_rate=0.0,
            seed=2026,
        )
        for mode in MODES
    }

    plain = datasets["plain"]
    masked = datasets["mask"]
    aged = datasets["mask_age"]

    assert len(plain) == len(masked) == len(aged)

    print(f"\n===== {split.upper()} =====")

    for r in range(len(plain.runs)):
        a = plain.observations[r]
        b = masked.observations[r]
        c = aged.observations[r]

        for key in (
            "observed_x",
            "mask",
            "age",
            "has_report",
            "source_time",
        ):
            assert torch.equal(
                a[key],
                b[key]
            ), key

            assert torch.equal(
                a[key],
                c[key]
            ), key

        T, N = a["mask"].shape

        now = torch.arange(
            1,
            T + 1
        ).unsqueeze(1)

        assert torch.all(
            a["source_time"] <= now
        )

        assert torch.equal(
            a["mask"].bool(),
            a["source_time"] == now
        )

        valid = a["has_report"].bool()

        assert torch.equal(
            a["age"][valid],
            (
                now.expand(T, N)
                - a["source_time"]
            )[valid].float()
        )

        for t in range(T):
            x_plain = plain.runs[r][t].x
            x_mask = masked.runs[r][t].x
            x_age = aged.runs[r][t].x

            expected = normalize_observed(
                a["observed_x"][t]
            )

            assert torch.allclose(
                x_plain,
                expected
            )

            assert torch.allclose(
                x_mask[:, :12],
                expected
            )

            assert torch.allclose(
                x_age[:, :12],
                expected
            )

            assert torch.equal(
                x_mask[:, 12],
                a["mask"][t]
            )

            assert torch.equal(
                x_age[:, 12],
                a["mask"][t]
            )

            expected_age = torch.where(
                valid[t],
                a["age"][t].clamp(
                    max=MAX_AGE
                ) / (MAX_AGE + 1),
                torch.ones_like(a["age"][t])
            )

            assert torch.allclose(
                x_age[:, 13],
                expected_age
            )

        actual_loss = (
            1 - a["mask"].mean().item()
        )

        print(
            f"Run {r + 1}: "
            f"Report loss={actual_loss:.2%} | "
            "Shared perturbation=PASSED | "
            "Mask/Age=PASSED"
        )

    assert (
        [w[2] for w in plain.windows]
        == [w[2] for w in masked.windows]
        == [w[2] for w in aged.windows]
    )

    print("Labels identical: PASSED")
    print("Causality: PASSED")


print("\nALL ABLATION DATASET CHECKS PASSED")