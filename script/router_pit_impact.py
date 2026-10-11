from pathlib import Path

import torch
from torch_geometric.data.separate import separate


ROOT = Path(
    "data/baseline_cache/non_existing/small/processed"
)

path = next(
    ROOT.glob("val_*_data.pt")
)

data, slices = torch.load(
    path,
    map_location="cpu",
    weights_only=False
)[:2]

routers = [
    "Rout4",
    "Rout1",
    "Rout9",
    "Rout6",
    "Rout2",
    "Rout7",
    "Rout5",
    "Rout8",
    "Rout3",
]

graphs = [
    separate(
        cls=data.__class__,
        batch=data,
        idx=i,
        slice_dict=slices,
        decrement=False
    )
    for i in range(60)
]

assert [
    int(g.time)
    for g in graphs
] == list(range(1, 61))

# Feature 0: PIT size.
pit = torch.stack([
    g.x[:, 0].float()
    for g in graphs
])

# Before IFA: t=41-50.
before = pit[40:50].mean(0)

# After IFA begins: t=51-60.
after = pit[50:60].mean(0)

delta = after - before

ranking = sorted(
    range(9),
    key=lambda i: float(delta[i]),
    reverse=True
)

print("\n===== ROUTER PIT IMPACT =====")

print(
    f"{'Router':<8} "
    f"{'Before':>10} "
    f"{'After':>10} "
    f"{'Delta':>10}"
)

print("-" * 43)

for i in ranking:
    print(
        f"{routers[i]:<8} "
        f"{before[i]:>10.2f} "
        f"{after[i]:>10.2f} "
        f"{delta[i]:>+10.2f}"
    )

print(
    "\nRouter with largest PIT increase:",
    routers[ranking[0]]
)

print("\nROUTER PIT AUDIT PASSED")