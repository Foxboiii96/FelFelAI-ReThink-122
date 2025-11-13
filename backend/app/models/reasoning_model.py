"""PyTorch-powered tiny reasoning model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from ..config import get_settings
from ..data.training_samples import TrainingSample, load_training_samples
from ..utils.text import build_vocabulary, tokenize_prompt


TASK_LABELS = {
    0: "addition",
    1: "parity",
    2: "reverse",
}


class TinyReasoningDataset(Dataset):
    """Torch dataset wrapping training samples."""

    def __init__(self, samples: Iterable[TrainingSample], vocab: Dict[str, int], max_length: int) -> None:
        self.samples = list(samples)
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:  # pragma: no cover - simple proxy
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        sample = self.samples[index]
        token_ids = tokenize_prompt(sample.prompt, self.vocab, self.max_length)
        x = torch.tensor(token_ids, dtype=torch.long)
        y = torch.tensor(sample.label, dtype=torch.long)
        return x, y


class TinyReasoningModel(nn.Module):
    """A lightweight character-level GRU classifier for task detection."""

    def __init__(self, vocab_size: int, embedding_dim: int = 32, hidden_dim: int = 64, num_classes: int = 3) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs: Tensor) -> Tensor:  # pragma: no cover - simple forwarding
        embeddings = self.embedding(inputs)
        _, hidden_state = self.gru(embeddings)
        logits = self.classifier(hidden_state.squeeze(0))
        return logits


class ReasoningEngine:
    """High level wrapper responsible for loading and running the model."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.device = torch.device("cpu")
        self.samples = load_training_samples()
        self.vocab = self._create_vocabulary(self.samples)
        self.model = TinyReasoningModel(len(self.vocab)).to(self.device)
        self._load_or_train()

    def _create_vocabulary(self, samples: Iterable[TrainingSample]) -> Dict[str, int]:
        chars = set("abcdefghijklmnopqrstuvwxyz0123456789?+-,. ")
        for sample in samples:
            chars.update(sample.prompt.lower())
        return build_vocabulary(chars)

    def _weights_metadata_path(self) -> Path:
        return self.settings.model_weights_path.with_suffix(".meta.json")

    def _load_or_train(self) -> None:
        weights_path = self.settings.model_weights_path
        metadata_path = self._weights_metadata_path()
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            return
        self._train_model(weights_path, metadata_path)

    def _train_model(self, weights_path: Path, metadata_path: Path) -> None:
        dataset = TinyReasoningDataset(self.samples, self.vocab, self.settings.max_sequence_length)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-3)
        self.model.train()
        for epoch in range(150):
            total_loss = 0.0
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                logits = self.model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 30 == 0:
                pass  # training is fast and silent
        self.model.eval()
        torch.save(self.model.state_dict(), weights_path)
        metadata = {
            "vocab": self.vocab,
            "max_sequence_length": self.settings.max_sequence_length,
            "task_labels": TASK_LABELS,
        }
        with metadata_path.open("w", encoding="utf-8") as meta_file:
            json.dump(metadata, meta_file, indent=2)

    def predict_task(self, prompt: str) -> Tuple[str, float]:
        """Return the predicted task label and the probability."""

        self.model.eval()
        token_ids = tokenize_prompt(prompt, self.vocab, self.settings.max_sequence_length)
        inputs = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(inputs)
            probabilities = torch.softmax(logits, dim=-1)
        confidence, label_idx = torch.max(probabilities, dim=-1)
        label_name = TASK_LABELS.get(label_idx.item(), "unknown")
        return label_name, confidence.item()


__all__ = ["ReasoningEngine", "TinyReasoningModel", "TASK_LABELS"]
