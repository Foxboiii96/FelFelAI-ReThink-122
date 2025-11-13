"""FastAPI application exposing the reasoning model."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .model import get_engine


class CompletionRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=512, description="Natural language instruction for the mini model")


class CompletionResponse(BaseModel):
    completion: str


app = FastAPI(title="Mini Reasoning LLM", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def warm_up_model() -> None:
    """Load the reasoning model into memory when the app boots."""
    get_engine()


@app.get("/", summary="Health check")
async def root() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/complete", response_model=CompletionResponse, summary="Generate a reasoning completion")
async def complete(request: CompletionRequest) -> CompletionResponse:
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Prompt must not be empty.")

    try:
        engine = get_engine()
        completion = engine.generate(prompt)
    except Exception as exc:  # pragma: no cover - defensive programming
        raise HTTPException(status_code=500, detail="Model failed to generate a response.") from exc

    if not completion:
        completion = "I am still learning and could not form a confident answer."

    return CompletionResponse(completion=completion)
