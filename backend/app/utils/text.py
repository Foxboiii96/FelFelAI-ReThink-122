"""Text processing utilities for the tiny reasoning model."""
from __future__ import annotations

import re
from typing import Dict, Iterable, List


def build_vocabulary(characters: Iterable[str]) -> Dict[str, int]:
    """Create a vocabulary mapping for a sequence of characters."""

    vocab = {char: idx + 2 for idx, char in enumerate(sorted(set(characters)))}
    vocab["<pad>"] = 0
    vocab["<unk>"] = 1
    return vocab


def tokenize_prompt(prompt: str, vocab: Dict[str, int], max_length: int) -> List[int]:
    """Convert a prompt into a fixed-length list of token ids."""

    prompt = prompt.lower()
    token_ids = [vocab.get(char, vocab["<unk>"]) for char in prompt]
    token_ids = token_ids[:max_length]
    if len(token_ids) < max_length:
        token_ids.extend([vocab["<pad>"]] * (max_length - len(token_ids)))
    return token_ids


def extract_numbers(prompt: str) -> List[int]:
    """Extract integers from a prompt string."""

    return [int(match) for match in re.findall(r"-?\d+", prompt)]


def extract_word_to_reverse(prompt: str) -> str | None:
    """Return the most likely word or phrase to reverse from the prompt."""

    match = re.search(r"reverse(?: the word)? ([a-zA-Z]+)", prompt, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"spell ([a-zA-Z]+) backwards", prompt, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"flip the string ([a-zA-Z]+)", prompt, re.IGNORECASE)
    if match:
        return match.group(1)
    words = re.findall(r"[a-zA-Z]+", prompt)
    return words[-1] if words else None


__all__ = [
    "build_vocabulary",
    "tokenize_prompt",
    "extract_numbers",
    "extract_word_to_reverse",
]
