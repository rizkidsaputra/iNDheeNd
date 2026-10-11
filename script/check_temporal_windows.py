from collections import Counter

from observed_temporal_dataset import (
    ObservedTemporalDataset,
    RUN_IDS,
    WINDOW,
)


EXPECTED = {
    "train": 882,
    "val": 294,
    "test": 294,
}


def main():
    print("===== TEMPORAL WINDOW AUDIT =====")

    for split, expected in EXPECTED.items():
        ds = ObservedTemporalDataset(
            split=split,
            mode="plain",
            loss_rate=0.0,
            delay_rate=0.0,
        )

        assert len(ds) == expected
        assert len(ds.runs) == len(RUN_IDS[split])

        labels = Counter()

        for run_index, last, label in ds.windows:
            assert 0 <= run_index < len(ds.runs)
            assert WINDOW - 1 <= last < len(
                ds.runs[run_index]
            )

            sequence = ds.runs[run_index][
                last - WINDOW + 1:last + 1
            ]

            assert len(sequence) == WINDOW
            assert label in (0, 1)

            labels[label] += 1

        # Each 298-timestamp run produces 294 windows.
        expected_normal = 46 * len(RUN_IDS[split])
        expected_ifa = 248 * len(RUN_IDS[split])

        assert labels[0] == expected_normal
        assert labels[1] == expected_ifa

        print(
            f"{split}: samples={len(ds)}, "
            f"normal={labels[0]}, IFA={labels[1]} "
            "| PASSED"
        )

    print("\nTEMPORAL WINDOW AUDIT PASSED")


if __name__ == "__main__":
    main()