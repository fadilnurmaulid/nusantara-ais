"""
Training pipeline for NusantaraAISModel.

Implements:
  - snapshot-level mini-batches via PyG HeteroData DataLoader (batch_size
    snapshots per step; heterogeneous disjoint-union batching is handled
    natively by PyG)
  - AMP (mixed precision) on CUDA, disabled automatically on CPU
  - gradient clipping (grad_clip_norm)
  - cosine LR scheduling
  - early stopping on validation total loss (patience = early_stopping_patience)
  - best-checkpoint saving (model + optimizer + config + epoch), enabling
    exact resume and the "model checkpointing" requirement
  - full per-epoch logging via the shared logger
"""
from __future__ import annotations

import copy
import os
import time
from typing import Dict, List

import torch
from torch_geometric.loader import DataLoader

from ..config import Config
from ..graph.dataset import MaritimeSnapshotDataset
from ..models.full_model import NusantaraAISModel
from ..utils.reproducibility import get_device, set_global_seed, setup_logger


def infer_node_feature_dims(sample_data) -> Dict[str, int]:
    return {ntype: sample_data[ntype].x.size(-1) for ntype in sample_data.node_types}


def train_model(cfg: Config, train_ds: MaritimeSnapshotDataset, val_ds: MaritimeSnapshotDataset,
                 checkpoint_name: str = "nusantara_ais_best.pt") -> Dict:
    logger = setup_logger("nusantara_ais.train", cfg.paths.abs(cfg.paths.logs_dir))
    set_global_seed(cfg.training.seed)
    device = get_device(cfg.training.device)
    logger.info(f"Training on device={device}, {len(train_ds)} train / {len(val_ds)} val snapshots")

    sample = train_ds.get(0)
    node_dims = infer_node_feature_dims(sample)
    edge_types = list(sample.edge_types)

    model = NusantaraAISModel(node_dims, edge_types, cfg.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs) \
        if cfg.training.lr_scheduler == "cosine" else None

    use_amp = cfg.training.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history: List[Dict] = []

    ckpt_dir = cfg.paths.abs(cfg.paths.checkpoint_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(cfg.training.epochs):
        t0 = time.time()
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss, _ = model(batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                loss, _ = model(batch)
                val_losses.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        train_loss = sum(train_losses) / max(len(train_losses), 1)
        val_loss = sum(val_losses) / max(len(val_losses), 1)
        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "time_s": elapsed})
        logger.info(f"epoch {epoch:03d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | {elapsed:.1f}s")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save({
                "model_state_dict": best_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": cfg.to_dict(),
                "node_feature_dims": node_dims,
                "edge_types": edge_types,
            }, os.path.join(ckpt_dir, checkpoint_name))
            logger.info(f"  -> new best checkpoint saved (val_loss={val_loss:.5f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.training.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch} (no improvement for "
                            f"{cfg.training.early_stopping_patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"model": model, "history": history, "best_val_loss": best_val_loss,
            "node_feature_dims": node_dims, "edge_types": edge_types, "device": device}
