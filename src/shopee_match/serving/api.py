"""FastAPI boundary for online and small-batch product matching."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, cast

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from shopee_match.errors import ContractError, DataValidationError, ShopeeMatchError
from shopee_match.serving.runtime import DemoRuntime, MatchPrediction

DEFAULT_CONFIG = Path("configs/serving/demo.yaml")


class CandidateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    posting_id: str
    title: str
    rank: int
    image_similarity: float | None
    title_similarity: float | None
    joint_similarity: float | None
    match_probability: float | None
    reciprocal_rank: int | None
    variant_conflict: bool | None
    accepted_match: bool | None
    decision_reason: str
    entity_id: str
    cluster_size: int
    cluster_confidence: float
    cluster_manual_review: bool


class MatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    query_mode: str
    query_posting_id: str | None
    predicted_entity_id: str | None
    confident_match: bool
    manual_review: bool
    decision_summary: str
    device: str
    index_backend: str
    candidate_k: int
    returned_k: int
    latency_ms: float
    candidates: list[CandidateResponse]


class BatchMatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    count: int
    predictions: list[MatchResponse]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    device: str
    index_backend: str
    catalog_split: str
    catalog_size: int
    candidate_k: int
    pair_probability_threshold: float
    supported_query_modes: list[str]
    ground_truth_used_for_inference: bool


class GuidedSampleResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    posting_id: str
    scenario: str
    description: str
    title: str
    display_title: str


def _runtime(app: FastAPI) -> DemoRuntime:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Inference runtime is not ready")
    return cast(DemoRuntime, runtime)


def _match_response(prediction: MatchPrediction) -> MatchResponse:
    return MatchResponse.model_validate(prediction.to_dict())


def create_app(
    runtime: DemoRuntime | None = None,
    *,
    config_path: Path | None = None,
) -> FastAPI:
    """Create an application; an injected runtime keeps API tests artifact-free."""
    resolved_config = config_path or Path(os.getenv("SHOPEE_DEMO_CONFIG", str(DEFAULT_CONFIG)))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if runtime is not None:
            app.state.runtime = runtime
        else:
            app.state.runtime = DemoRuntime.load(resolved_config)
        yield

    app = FastAPI(
        title="Shopee Multimodal Product Matching",
        version="1.0.0",
        description=(
            "Frozen validation-catalog demonstration of retrieval, pair scoring, and entity "
            "resolution. Ground-truth labels are not used by inference."
        ),
        lifespan=lifespan,
    )

    @app.exception_handler(ContractError)
    async def contract_error_handler(_request: Any, exc: ContractError) -> Any:
        return _json_error(400, str(exc))

    @app.exception_handler(DataValidationError)
    async def data_error_handler(_request: Any, exc: DataValidationError) -> Any:
        return _json_error(503, str(exc))

    @app.get("/health", response_model=HealthResponse, tags=["operations"])
    def health() -> dict[str, Any]:
        return _runtime(app).health()

    @app.get(
        "/api/v1/guided-samples",
        response_model=list[GuidedSampleResponse],
        tags=["catalog"],
    )
    def guided_samples() -> list[dict[str, str]]:
        return _runtime(app).guided_samples()

    @app.post("/api/v1/match", response_model=MatchResponse, tags=["matching"])
    async def match(
        image: Annotated[UploadFile | None, File(description="Optional product image")] = None,
        title: Annotated[str | None, Form(max_length=500)] = None,
        top_k: Annotated[int | None, Form(ge=1)] = None,
        query_posting_id: Annotated[str | None, Form()] = None,
    ) -> MatchResponse:
        service = _runtime(app)
        content = (
            await image.read(service.config.runtime.maximum_upload_bytes + 1)
            if image is not None
            else None
        )
        return _match_response(
            service.match(content, title, top_k, query_posting_id=query_posting_id)
        )

    @app.post("/api/v1/match/batch", response_model=BatchMatchResponse, tags=["matching"])
    async def match_batch(
        images: Annotated[list[UploadFile], File(description="One image per title")],
        titles: Annotated[list[str], Form()],
        top_k: Annotated[int | None, Form(ge=1)] = None,
    ) -> BatchMatchResponse:
        service = _runtime(app)
        if len(images) != len(titles):
            raise HTTPException(status_code=400, detail="images and titles must have equal length")
        if not images or len(images) > service.config.runtime.maximum_batch_size:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"batch size must be between 1 and {service.config.runtime.maximum_batch_size}"
                ),
            )
        predictions = []
        for image, title in zip(images, titles, strict=True):
            content = await image.read(service.config.runtime.maximum_upload_bytes + 1)
            predictions.append(_match_response(service.match(content, title, top_k)))
        return BatchMatchResponse(
            status="complete", count=len(predictions), predictions=predictions
        )

    @app.get("/api/v1/catalog/{posting_id}/image", tags=["catalog"])
    def catalog_image(posting_id: str) -> FileResponse:
        try:
            path = _runtime(app).catalog_image_path(posting_id)
        except ShopeeMatchError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path)

    return app


def _json_error(status_code: int, detail: str) -> Any:
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=status_code, content={"detail": detail})
