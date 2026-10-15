import sys
from pathlib import Path

import torch
from torch_geometric.nn import SAGEConv
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path.cwd() / "src"))

from temporal_dataset import NDNTemporalGraphDataset

data_dir = Path("data/processed/ml2")

run_ids = sorted(
    p.stem for p in data_dir.glob("*.npz")
)

dataset = NDNTemporalGraphDataset(
    data_dir=data_dir,
    run_ids=run_ids
)

assert len(dataset) == 276

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

sage = SAGEConv(10, 64).to(device)

gru = torch.nn.GRU(
    input_size=64,
    hidden_size=64,
    batch_first=True
).to(device)

sage.eval()
gru.eval()

batch = next(iter(loader))

x = batch["x"].to(device)
edge_index = dataset.edge_index.to(device)

B, T, N, F = x.shape

assert (T, N, F) == (5, 10, 10)

graph_embeddings = []

with torch.no_grad():
    for t in range(T):
        node_embeddings = []

        for b in range(B):
            z = sage(
                x[b, t],
                edge_index
            )

            # Temporary global mean pooling.
            pooled = z.mean(dim=0)
            node_embeddings.append(pooled)

        graph_embeddings.append(
            torch.stack(node_embeddings)
        )

    sequence = torch.stack(
        graph_embeddings,
        dim=1
    )

    output, hidden = gru(sequence)

assert sequence.shape == (B, 5, 64)
assert output.shape == (B, 5, 64)
assert hidden.shape == (1, B, 64)
assert torch.isfinite(output).all()

print("Dataset windows:", len(dataset))
print("Batch X:", tuple(x.shape))
print("Graph embeddings:", tuple(sequence.shape))
print("GRU output:", tuple(output.shape))
print("Device:", device)
print("ML-2 loader smoke test: PASSED")
