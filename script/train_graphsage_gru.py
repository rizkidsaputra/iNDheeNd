import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.data.separate import separate
from torch_geometric.nn import SAGEConv, global_mean_pool

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(
    "data/baseline_cache/non_existing/small/processed"
)

OUT = Path("results/graphsage_gru_clean_pilot")

WINDOW = 5
SEED = 42
EPOCHS = 30
BATCH_SIZE = 16

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

OUT.mkdir(parents=True, exist_ok=True)


class TemporalDataset(Dataset):
    """Temporal dataset for complete monitoring observations."""

    def __init__(self, split):
        path = next(ROOT.glob(f"{split}_*_data.pt"))

        # Load only locally generated trusted PyG cache.
        obj = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        data, slices = obj[:2]
        count = len(slices["x"]) - 1

        self.graphs = []
        times = []
        labels = []

        for i in range(count):
            g = separate(
                cls=data.__class__,
                batch=data,
                idx=i,
                slice_dict=slices,
                decrement=False,
            )

            x = g.x.float()

            # Pilot normalization.
            x = x - x.min()
            x = x / x.sum(
                dim=-1,
                keepdim=True
            ).clamp(min=1.0)

            assert x.shape == (9, 12)
            assert torch.isfinite(x).all()

            self.graphs.append(
                Data(
                    x=x,
                    edge_index=g.edge_index.long(),
                )
            )

            times.append(int(g.time))
            labels.append(int(g.attack_is_on))

        # A new simulation run begins when time resets.
        boundaries = [0] + [
            i for i in range(1, count)
            if times[i] <= times[i - 1]
        ] + [count]

        self.windows = []
        self.labels = []

        for start, end in zip(
            boundaries[:-1],
            boundaries[1:]
        ):
            run_times = times[start:end]

            assert all(
                b == a + 1
                for a, b in zip(
                    run_times[:-1],
                    run_times[1:]
                )
            )

            for last in range(
                start + WINDOW - 1,
                end
            ):
                first = last - WINDOW + 1

                self.windows.append(
                    list(range(first, last + 1))
                )

                self.labels.append(labels[last])

        print(
            f"{split}: {len(self.windows)} windows, "
            f"normal={self.labels.count(0)}, "
            f"IFA={self.labels.count(1)}"
        )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        graphs = [
            self.graphs[i]
            for i in self.windows[idx]
        ]

        return graphs, self.labels[idx]


def collate_temporal(samples):
    """
    Create one PyG Batch per temporal position.
    """
    batches = [
        Batch.from_data_list(
            [sample[0][t] for sample in samples]
        )
        for t in range(WINDOW)
    ]

    labels = torch.tensor(
        [sample[1] for sample in samples],
        dtype=torch.long,
    )

    return batches, labels


class GraphSAGEGRU(nn.Module):

    def __init__(
        self,
        input_dim=12,
        hidden_dim=64,
    ):
        super().__init__()

        self.sage1 = SAGEConv(
            input_dim,
            hidden_dim
        )

        self.sage2 = SAGEConv(
            hidden_dim,
            hidden_dim
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Linear(
            hidden_dim,
            2
        )

    def forward(self, graph_batches):
        temporal_embeddings = []

        for graph in graph_batches:
            x = F.relu(
                self.sage1(
                    graph.x,
                    graph.edge_index
                )
            )

            x = F.relu(
                self.sage2(
                    x,
                    graph.edge_index
                )
            )

            embedding = global_mean_pool(
                x,
                graph.batch
            )

            temporal_embeddings.append(embedding)

        sequence = torch.stack(
            temporal_embeddings,
            dim=1
        )

        _, hidden = self.gru(sequence)

        return self.classifier(
            self.dropout(hidden[-1])
        )


def evaluate_predictions(y_true, y_pred):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred
        ),
        "precision_ifa": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall_ifa": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1_ifa": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "false_positive_rate": (
            float(fp / (fp + tn))
            if (fp + tn) else 0.0
        ),
        "confusion_matrix": cm.tolist(),
    }


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0

    y_true = []
    y_pred = []

    for batches, labels in loader:
        batches = [
            graph.to(device)
            for graph in batches
        ]

        labels = labels.to(device)

        with torch.set_grad_enabled(training):
            logits = model(batches)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += (
            loss.item() * labels.size(0)
        )

        y_true.extend(
            labels.cpu().tolist()
        )

        y_pred.extend(
            logits.argmax(dim=1).cpu().tolist()
        )

    metrics = evaluate_predictions(
        y_true,
        y_pred
    )

    metrics["loss"] = (
        total_loss / len(loader.dataset)
    )

    metrics["n_samples"] = len(y_true)

    return metrics


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    train_data = TemporalDataset("train")
    val_data = TemporalDataset("val")
    test_data = TemporalDataset("test")

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_temporal,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_temporal,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_temporal,
    )

    model = GraphSAGEGRU().to(device)

    counts = torch.bincount(
        torch.tensor(train_data.labels),
        minlength=2,
    ).float()

    weights = counts.sum() / (
        2.0 * counts.clamp(min=1)
    )

    criterion = nn.CrossEntropyLoss(
        weight=weights.to(device)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=5e-4,
    )

    best_loss = float("inf")
    checkpoint_path = OUT / "best_model.pt"

    for epoch in range(1, EPOCHS + 1):
        train = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )

        val = run_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        if val["loss"] < best_loss:
            best_loss = val["loss"]

            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "window": WINDOW,
                "hidden_dim": 64,
                "seed": SEED,
            }, checkpoint_path)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss={train['loss']:.4f} "
            f"F1={train['f1_ifa']:.4f} | "
            f"Val loss={val['loss']:.4f} "
            f"F1={val['f1_ifa']:.4f} "
            f"BalancedAcc="
            f"{val['balanced_accuracy']:.4f}"
        )

    saved = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        saved["model_state_dict"]
    )

    val = run_epoch(
        model,
        val_loader,
        criterion,
        device
    )

    test = run_epoch(
        model,
        test_loader,
        criterion,
        device
    )

    results = {
        "model": "GraphSAGE-GRU",
        "dataset": (
            "SPOTIFAI pilot "
            "non_existing/small/16x/4_attackers"
        ),
        "window": WINDOW,
        "best_epoch": saved["epoch"],
        "validation": val,
        "test": test,
    }

    with open(OUT / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n===== FINAL VALIDATION =====")
    print(json.dumps(val, indent=2))

    print("\n===== FINAL TEST =====")
    print(json.dumps(test, indent=2))

    print("\nSaved to:", OUT)


if __name__ == "__main__":
    main()