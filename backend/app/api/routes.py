"""API routes for the reasoning service."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..schemas import CompletionRequest, CompletionResponse
from ..services.reasoning_service import ReasoningService

router = APIRouter()
service = ReasoningService()


@router.post("/complete", response_model=CompletionResponse, status_code=status.HTTP_200_OK)
async def create_completion(request: CompletionRequest) -> CompletionResponse:
    """Return a completion produced by the tiny reasoning model."""

    try:
        result = service.complete(request.prompt)
    except Exception as exc:  # pragma: no cover - defensive programming
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return CompletionResponse(completion=result.completion)


__all__ = ["router"]
