"""Reasoning model powered by a tiny Seq2Seq network."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch import nn


SPECIAL_TOKENS = {
    "<pad>": 0,
    "<sos>": 1,
    "<eos>": 2,
    "<unk>": 3,
}


class SimpleTokenizer:
    """A minimalist tokenizer that keeps digits and words separate."""

    def __init__(self) -> None:
        self.token_to_idx = dict(SPECIAL_TOKENS)
        self.idx_to_token = {idx: tok for tok, idx in SPECIAL_TOKENS.items()}

    _token_pattern = re.compile(r"\d+|[A-Za-z']+|[^\w\s]")

    def tokenize(self, text: str) -> List[str]:
        tokens = self._token_pattern.findall(text.lower())
        return tokens

    def add_sentence(self, text: str) -> None:
        for token in self.tokenize(text):
            self.add_token(token)

    def add_token(self, token: str) -> None:
        if token not in self.token_to_idx:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token

    def encode(self, text: str) -> List[int]:
        tokens = [self.token_to_idx.get(token, self.unk_idx) for token in self.tokenize(text)]
        return tokens

    def decode(self, token_ids: Sequence[int]) -> str:
        tokens = []
        for idx in token_ids:
            if idx == SPECIAL_TOKENS["<eos>"]:
                break
            if idx in (SPECIAL_TOKENS["<sos>"], SPECIAL_TOKENS["<pad>"]):
                continue
            tokens.append(self.idx_to_token.get(idx, ""))
        text = " ".join(tokens)
        text = text.replace(" ,", ",").replace(" .", ".")
        text = text.replace(" ?", "?").replace("' ", "'")
        return text.strip()

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_idx)

    @property
    def pad_idx(self) -> int:
        return SPECIAL_TOKENS["<pad>"]

    @property
    def sos_idx(self) -> int:
        return SPECIAL_TOKENS["<sos>"]

    @property
    def eos_idx(self) -> int:
        return SPECIAL_TOKENS["<eos>"]

    @property
    def unk_idx(self) -> int:
        return SPECIAL_TOKENS["<unk>"]


class TinySeq2Seq(nn.Module):
    """A compact GRU-based encoder-decoder."""

    def __init__(self, vocab_size: int, embed_dim: int = 96, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=SPECIAL_TOKENS["<pad>"])
        self.decoder_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=SPECIAL_TOKENS["<pad>"])
        self.encoder_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.decoder_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.output_linear = nn.Linear(hidden_dim, vocab_size)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        embedded = self.encoder_embed(src)
        _, hidden = self.encoder_gru(embedded)
        return hidden

    def decode_step(self, input_token: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.decoder_embed(input_token.unsqueeze(1))
        output, hidden = self.decoder_gru(embedded, hidden)
        logits = self.output_linear(output.squeeze(1))
        return logits, hidden


@dataclass
class TrainingSample:
    prompt: str
    completion: str


class ReasoningEngine:
    """Builds, trains, and serves a tiny reasoning model."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cpu")
        random.seed(42)
        torch.manual_seed(42)

        self.dataset = self._generate_dataset()
        self.tokenizer = SimpleTokenizer()

        for sample in self.dataset:
            self.tokenizer.add_sentence(sample.prompt)
            self.tokenizer.add_sentence(sample.completion)

        self.model = TinySeq2Seq(self.tokenizer.vocab_size)
        self.model.to(self.device)

        self._train()

    def _generate_dataset(self) -> List[TrainingSample]:
        samples: List[TrainingSample] = []
        for a in range(0, 10):
            for b in range(0, 10):
                sum_val = a + b
                samples.append(
                    TrainingSample(
                        prompt=f"what is {a} plus {b}?",
                        completion=f"first, compute {a} + {b} = {sum_val}. therefore the answer is {sum_val}.",
                    )
                )
                samples.append(
                    TrainingSample(
                        prompt=f"add {a} and {b}.",
                        completion=f"adding {a} and {b} gives {sum_val}. so the final answer is {sum_val}.",
                    )
                )
                diff = a - b
                samples.append(
                    TrainingSample(
                        prompt=f"what is {a} minus {b}?",
                        completion=f"subtracting {b} from {a} gives {diff}. the answer is {diff}.",
                    )
                )
                bigger = a if a >= b else b
                smaller = b if a >= b else a
                comparison = (
                    f"comparing {a} and {b}, {bigger} is larger than {smaller}. so the answer is {bigger}."
                )
                samples.append(
                    TrainingSample(prompt=f"which number is larger, {a} or {b}?", completion=comparison)
                )
                if a < 9:
                    next_val = a + 1
                    samples.append(
                        TrainingSample(
                            prompt=f"what comes after {a}?",
                            completion=f"counting forward from {a} leads to {next_val}. so the answer is {next_val}.",
                        )
                    )
                parity = "even" if a % 2 == 0 else "odd"
                samples.append(
                    TrainingSample(
                        prompt=f"is {a} even or odd?",
                        completion=f"{a} is {parity} because it is divisible by 2." if parity == "even" else f"{a} is odd because it leaves a remainder of 1 when divided by 2.",
                    )
                )
        random.shuffle(samples)
        return samples

    def _tensorize(self, text: str) -> torch.Tensor:
        tokens = [self.tokenizer.sos_idx, *self.tokenizer.encode(text), self.tokenizer.eos_idx]
        tensor = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        return tensor

    def _train(self) -> None:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(30):
            total_loss = 0.0
            random.shuffle(self.dataset)
            for sample in self.dataset:
                src = self._tensorize(sample.prompt)
                tgt = self._tensorize(sample.completion)

                optimizer.zero_grad()
                hidden = self.model.encode(src)

                input_token = torch.tensor([self.tokenizer.sos_idx], dtype=torch.long, device=self.device)
                loss = 0.0
                for idx in range(1, tgt.size(1)):
                    logits, hidden = self.model.decode_step(input_token, hidden)
                    target = tgt[:, idx]
                    loss_step = criterion(logits, target)
                    loss = loss + loss_step
                    input_token = tgt[:, idx]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0:
                avg_loss = total_loss / max(len(self.dataset), 1)
                # Print loss occasionally for debug (not required but helpful during dev)
                print(f"[ReasoningEngine] epoch={epoch} loss={avg_loss:.4f}")

        self.model.eval()

    def generate(self, prompt: str, max_length: int = 64) -> str:
        self.model.eval()
        with torch.no_grad():
            src = self._tensorize(prompt)
            hidden = self.model.encode(src)
            input_token = torch.tensor([self.tokenizer.sos_idx], dtype=torch.long, device=self.device)
            generated: List[int] = []
            for _ in range(max_length):
                logits, hidden = self.model.decode_step(input_token, hidden)
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1)
                token_id = next_token.item()
                if token_id == self.tokenizer.eos_idx:
                    break
                generated.append(token_id)
                input_token = next_token
            return self.tokenizer.decode(generated)


# Singleton engine used by the API layer
_engine: ReasoningEngine | None = None


def get_engine() -> ReasoningEngine:
    global _engine
    if _engine is None:
        _engine = ReasoningEngine()
    return _engine
