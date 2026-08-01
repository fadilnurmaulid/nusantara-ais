"""
PyTorch Geometric dataset wrapper.

Wraps the list of per-snapshot `HeteroData` objects produced by
`MaritimeGraphBuilder` in a `torch_geometric.data.Dataset`, enabling
standard PyG `DataLoader` batching (heterogeneous graphs of different
snapshots are batched by disjoint union, which is memory-efficient and the
canonical PyG pattern).

Split strategy
--------------
Snapshots are split chronologically (NOT randomly): train = earliest
`1 - val_split - test_split` fraction, val = next `val_split`, test = the
final `test_split`. A random split would leak future spatial/behavioral
context into training via the temporal_next and spatial_proximity edges of
adjacent snapshots, inflating validation performance -- chronological
splitting is the only leak-free choice for a temporally-snapshotted graph
dataset and mirrors real deployment (train on the past, evaluate on the
future).
"""
from __future__ import annotations

from typing import List, Tuple

from torch_geometric.data import Dataset, HeteroData

from ..config import TrainingConfig


class MaritimeSnapshotDataset(Dataset):
    def __init__(self, snapshots: List[HeteroData]):
        super().__init__()
        self.snapshots = snapshots

    def len(self) -> int:
        return len(self.snapshots)

    def get(self, idx: int) -> HeteroData:
        return self.snapshots[idx]


def chronological_split(snapshots: List[HeteroData], train_cfg: TrainingConfig
                         ) -> Tuple[MaritimeSnapshotDataset, MaritimeSnapshotDataset, MaritimeSnapshotDataset]:
    n = len(snapshots)
    n_test = max(1, int(n * train_cfg.test_split))
    n_val = max(1, int(n * train_cfg.val_split))
    n_train = max(1, n - n_val - n_test)

    train = snapshots[:n_train]
    val = snapshots[n_train:n_train + n_val]
    test = snapshots[n_train + n_val:]
    return (MaritimeSnapshotDataset(train), MaritimeSnapshotDataset(val), MaritimeSnapshotDataset(test))
