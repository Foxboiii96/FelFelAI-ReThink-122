"""Synthetic training samples for the tiny reasoning model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TrainingSample:
    """A single training example."""

    prompt: str
    label: int


def load_training_samples() -> List[TrainingSample]:
    """Return a curated list of prompts and their associated task labels."""

    addition_prompts = [
        "What is 2 plus 3?",
        "Add 4 and 5 step by step.",
        "Please compute 7 + 2 with reasoning.",
        "Can you explain 3 plus 6?",
        "Walk me through adding 1 and 8.",
        "Figure out the sum of 9 and 0.",
        "How much is 5 added to 4?",
        "What do you get by summing 6 with 3?",
        "Explain how to add 2 and 4.",
        "Need a chain of thought for 1 + 5.",
        "Solve 8 plus 1 carefully.",
    ]

    parity_prompts = [
        "Is 4 even or odd?",
        "Determine if 9 is an even number.",
        "Tell me whether 7 should be odd.",
        "Reason about the parity of 2.",
        "Explain if 5 counts as even.",
        "Is 10 considered odd or even?",
        "How would you classify the number 3?",
        "Check if 12 is even.",
        "Evaluate parity for 1.",
        "Is 0 even?",
        "Give me the parity of 15.",
    ]

    reverse_prompts = [
        "Reverse the word apple.",
        "Can you spell world backwards?",
        "Flip the string hello.",
        "Provide the reverse of robot.",
        "Backwards version of tiny please.",
        "Reverse order for llama.",
        "Spell magic in reverse order.",
        "What is smart reversed?",
        "Show me the backward spelling of code.",
        "How do you reverse reason?",
        "Turn lights backward.",
    ]

    samples: List[TrainingSample] = []

    samples.extend(TrainingSample(prompt, 0) for prompt in addition_prompts)
    samples.extend(TrainingSample(prompt, 1) for prompt in parity_prompts)
    samples.extend(TrainingSample(prompt, 2) for prompt in reverse_prompts)

    return samples


__all__ = ["TrainingSample", "load_training_samples"]
