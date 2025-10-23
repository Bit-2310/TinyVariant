from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from models.losses import IGNORE_LABEL_ID


def _compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))

    pos = labels == 1
    neg = labels == 0
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    auc = (ranks[pos].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)
    return float(auc)


class ClinVarEvaluator:
    required_outputs = {"logits"}

    def __init__(
        self,
        data_path: str,
        eval_metadata,
        vocab_path: Optional[str] = None,
        positive_label: str = "LABEL_PATHOGENIC",
        negative_label: str = "LABEL_BENIGN",
    ):
        path = Path(vocab_path) if vocab_path is not None else Path(data_path) / "vocab.json"
        vocab_items = json.loads(path.read_text())
        vocab = {token: idx for token, idx in vocab_items}
        self.pos_id = vocab[positive_label]
        self.neg_id = vocab[negative_label]
        self._scores = []
        self._labels = []
        self._preds = []

    def begin_eval(self):
        self._scores = []
        self._labels = []
        self._preds = []

    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        logits = preds["logits"].cpu()
        labels = batch["labels"].cpu()
        mask = labels[:, -1] != IGNORE_LABEL_ID
        if not mask.any():
            return

        logits = logits[mask][:, -1, :]
        labels = labels[mask][:, -1].to(torch.long)

        probs = torch.softmax(logits, dim=-1)
        self._scores.extend(probs[:, self.pos_id].tolist())
        self._labels.extend(labels.tolist())
        self._preds.extend(logits.argmax(dim=-1).tolist())

    def result(
        self,
        save_path: Optional[str],
        rank: int,
        world_size: int,
        group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> Optional[Dict[str, float]]:
        if rank != 0:
            return None

        scores = np.array(self._scores, dtype=np.float64)
        labels = np.array(self._labels, dtype=np.int64)
        preds = np.array(self._preds, dtype=np.int64)
        if labels.size == 0:
            return {"ClinVar/accuracy": float("nan"), "ClinVar/roc_auc": float("nan")}

        label_binary = (labels == self.pos_id).astype(np.int64)
        accuracy = float((preds == labels).mean())

        auc = _compute_auc(label_binary, scores)
        return {"ClinVar/accuracy": accuracy, "ClinVar/roc_auc": auc}
