from pathlib import Path

import torch
from torch_geometric.data.separate import separate


ROOT = Path(
    "data/baseline_cache/non_existing/small/processed"
)

ROUTER_MAPPING = {
    0: "Rout4",
    1: "Rout1",
    2: "Rout9",
    3: "Rout6",
    4: "Rout2",
    5: "Rout7",
    6: "Rout5",
    7: "Rout8",
    8: "Rout3",
}

EXPECTED = {
    "train": 894,
    "val": 298,
    "test": 298,
}


def main():
    reference_edges = None

    for split, expected_count in EXPECTED.items():
        path = next(
            ROOT.glob(f"{split}_*_data.pt")
        )

        data, slices = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )[:2]

        count = len(slices["x"]) - 1
        assert count == expected_count

        for i in range(count):
            graph = separate(
                cls=data.__class__,
                batch=data,
                idx=i,
                slice_dict=slices,
                decrement=False
            )

            assert graph.x.shape == (9, 12)
            assert graph.edge_index.shape == (2, 10)

            if reference_edges is None:
                reference_edges = (
                    graph.edge_index.clone()
                )
            else:
                assert torch.equal(
                    graph.edge_index,
                    reference_edges
                )

        print(
            f"{split}: {count} graphs | "
            "shape and topology PASSED"
        )

    print("\nPILOT ROUTER MAPPING")

    for index, name in ROUTER_MAPPING.items():
        print(f"Node {index}: {name}")

    print("\nPILOT GRAPH TOPOLOGY CHECK PASSED")
    print(
        "Note: This does not independently "
        "revalidate router names against raw PIT traces."
    )


if __name__ == "__main__":
    main()