"""Service layer orchestrating model inference and reasoning."""
from __future__ import annotations

from dataclasses import dataclass

from ..utils.text import extract_numbers, extract_word_to_reverse
from ..models.reasoning_model import ReasoningEngine


@dataclass
class ReasoningResult:
    """Container for reasoning output."""

    completion: str


class ReasoningService:
    """High level orchestration between the ML model and heuristics."""

    def __init__(self) -> None:
        self.engine = ReasoningEngine()

    def complete(self, prompt: str) -> ReasoningResult:
        task, confidence = self.engine.predict_task(prompt)

        if task == "addition":
            completion = self._handle_addition(prompt, confidence)
        elif task == "parity":
            completion = self._handle_parity(prompt, confidence)
        elif task == "reverse":
            completion = self._handle_reverse(prompt, confidence)
        else:
            completion = self._handle_unknown(prompt)

        return ReasoningResult(completion=completion)

    def _handle_addition(self, prompt: str, confidence: float) -> str:
        numbers = extract_numbers(prompt)
        if len(numbers) < 2:
            return (
                "I was expecting an addition problem but couldn't locate two numbers. "
                "Try phrasing it like 'add 2 and 3'."
            )
        total = sum(numbers[:2])
        return (
            "Let's solve this addition carefully.\n"
            f"1. Identify the numbers: {numbers[0]} and {numbers[1]}.\n"
            f"2. Add them together: {numbers[0]} + {numbers[1]} = {total}.\n"
            f"3. Therefore, the final answer is {total}.\n"
            f"(model confidence in task type: {confidence:.2%})"
        )

    def _handle_parity(self, prompt: str, confidence: float) -> str:
        numbers = extract_numbers(prompt)
        if not numbers:
            return "I expected a number to check parity, but none was provided."
        number = numbers[0]
        is_even = number % 2 == 0
        parity = "even" if is_even else "odd"
        explanation = (
            "To determine parity, we check divisibility by 2.\n"
            f"1. Start with the number {number}.\n"
            f"2. {number} % 2 = {number % 2}.\n"
            f"3. Because the remainder is {'0' if is_even else '1'}, the number is {parity}.\n"
            f"Result: {number} is {parity}.\n"
            f"(model confidence in task type: {confidence:.2%})"
        )
        return explanation

    def _handle_reverse(self, prompt: str, confidence: float) -> str:
        target = extract_word_to_reverse(prompt)
        if not target:
            return "I wanted to reverse a word, but I couldn't find one in the prompt."
        reversed_word = target[::-1]
        return (
            "Reversing the word is straightforward.\n"
            f"1. Take the word '{target}'.\n"
            "2. Read the characters from the end to the beginning.\n"
            f"3. The reversed form is '{reversed_word}'.\n"
            f"(model confidence in task type: {confidence:.2%})"
        )

    def _handle_unknown(self, prompt: str) -> str:
        return (
            "I am a tiny reasoning model trained for quick arithmetic, parity checks, "
            "and word reversals. Please ask about one of those topics!"
        )


__all__ = ["ReasoningService", "ReasoningResult"]
