import logging
import time

from auth.database import init_db
from auth.router import router as auth_router
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.pipeline import Pipeline
from app.pipeline.autoencoder import AutoEncoderPipeline
from app.pipeline.bert import EnsembleBERTPipeline
from app.pipeline.llm_judge import LLM_JudgePipeline
from app.pipeline.semantic_search import SemanticSearchPipeline
from app.type_store import Phase, PhaseInput, Verdict

logger = logging.getLogger(__name__)

# a verdict from autoencoder / bert / llm_judge must reach this confidence
# before it is written back into the semantic search index
CONFIRM_CONFIDENCE_THRESHOLD = 0.90

app = FastAPI()

# built once at startup (see on_startup below), not per-request - the
# autoencoder and bert stages load a model into memory, so we don't want to
# reload them on every call to /screen.
detection_pipeline: Pipeline | None = None
semantic_search_phase: SemanticSearchPipeline | None = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()

    global detection_pipeline
    # kept as its own reference so /screen can write confirmed verdicts back
    # into its index via .confirm()
    global semantic_search_phase
    semantic_search_phase = SemanticSearchPipeline()
    detection_pipeline = Pipeline(
        [
            semantic_search_phase,
            AutoEncoderPipeline(),
            EnsembleBERTPipeline(),
            LLM_JudgePipeline(),
        ]
    )


app.include_router(auth_router)


@app.get("/")
def home():
    return {"message": "Backend running!"}


class ScreenRequest(BaseModel):
    prompt: str


class ScreenResponse(BaseModel):
    verdict: str  # "benign" or "attack" (never "undetermined" - the last stage always decides)
    phase: str  # which stage in the cascade produced this verdict
    confidence: float
    latency_ms: float  # time from input received to verdict given, summed across every stage that ran


@app.post("/screen", response_model=ScreenResponse)
def screen(request: ScreenRequest):
    if detection_pipeline is None:
        raise HTTPException(
            status_code=503, detail="detection pipeline is not ready yet"
        )

    start_time = time.perf_counter()
    result = detection_pipeline.run(PhaseInput(text=request.prompt))

    if result.is_err():
        # a stage broke - fail closed rather than letting the prompt through
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return ScreenResponse(
            verdict=Verdict.attack.name,
            phase="error",
            confidence=0.0,
            latency_ms=elapsed_ms,
        )

    success = result.unwrap()
    response = ScreenResponse(
        verdict=success.verdict.name,
        phase=success.at_phase.name,
        confidence=success.confidence,
        latency_ms=success.latency_ms,
    )

    # self-improving write-back: only for later stages (semantic search already
    # has this prompt) and only when that stage was confident. Never allowed to
    # break or delay the response, so failures are just logged.
    if (
        semantic_search_phase is not None
        and success.at_phase != Phase.semantic_search
        and success.confidence >= CONFIRM_CONFIDENCE_THRESHOLD
    ):
        try:
            label = "attack" if success.verdict == Verdict.attack else "benign"
            semantic_search_phase.confirm(request.prompt, label)
        except Exception as e:
            logger.warning("semantic search write-back failed: %s", e)

    return response
