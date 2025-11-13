"""Pydantic schemas used by the API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    """Input schema for the completion endpoint."""

    prompt: str = Field(..., min_length=1, max_length=512, description="User prompt to complete")


class CompletionResponse(BaseModel):
    """Output schema for the completion endpoint."""

    completion: str


__all__ = ["CompletionRequest", "CompletionResponse"]
