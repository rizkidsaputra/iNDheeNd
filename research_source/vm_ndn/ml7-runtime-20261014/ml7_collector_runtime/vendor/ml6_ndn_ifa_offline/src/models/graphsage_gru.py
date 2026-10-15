import torch
from torch import nn
from torch_geometric.nn import SAGEConv


class GraphSAGEGRU(nn.Module):
    """
    Binary NDN IFA classifier.

    Input:
        x: [batch, time, nodes, features]
        edge_index: [2, directed_edges]

    Output:
        logits: [batch]

    No Mask/Age input yet (ML-3 Plain baseline).
    """

    def __init__(
        self,
        input_dim=10,
        hidden_dim=64,
        gru_hidden_dim=64,
        dropout=0.2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv1 = SAGEConv(
            input_dim,
            hidden_dim
        )

        self.conv2 = SAGEConv(
            hidden_dim,
            hidden_dim
        )

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.classifier = nn.Linear(
            gru_hidden_dim,
            1
        )

    def forward(self, x, edge_index):
        if x.ndim != 4:
            raise ValueError(
                "Expected x shape [B,T,N,F]"
            )

        B, T, N, F = x.shape

        if F != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features, got {F}"
            )

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("Invalid edge_index")

        if edge_index.device != x.device:
            edge_index = edge_index.to(x.device)

        if not torch.isfinite(x).all():
            raise ValueError("Non-finite model input")

        # Each graph is disconnected from graphs
        # belonging to other samples or timesteps.
        graph_count = B * T
        E = edge_index.shape[1]

        offsets = (
            torch.arange(
                graph_count,
                device=x.device,
                dtype=edge_index.dtype
            )
            .repeat_interleave(E)
            * N
        )

        batched_edges = (
            edge_index.repeat(1, graph_count)
            + offsets.unsqueeze(0)
        )

        nodes = x.reshape(
            graph_count * N,
            F
        )

        h = self.conv1(nodes, batched_edges)
        h = self.activation(h)
        h = self.dropout(h)

        h = self.conv2(h, batched_edges)
        h = self.activation(h)
        h = self.dropout(h)

        # Global mean pooling per timestep.
        graph_embeddings = h.reshape(
            B, T, N, self.hidden_dim
        ).mean(dim=2)

        temporal_output, _ = self.gru(
            graph_embeddings
        )

        final_embedding = temporal_output[:, -1, :]

        logits = self.classifier(
            final_embedding
        ).squeeze(-1)

        return logits

    @torch.no_grad()
    def predict_proba(self, x, edge_index):
        self.eval()
        logits = self.forward(x, edge_index)
        return torch.sigmoid(logits)
