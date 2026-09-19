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
from app.type_store import PhaseInput, Verdict

app = FastAPI()

# built once at startup (see on_startup below), not per-request - the
# autoencoder and bert stages load a model into memory, so we don't want to
# reload them on every call to /screen.
detection_pipeline: Pipeline | None = None

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
    detection_pipeline = Pipeline(
        [
            SemanticSearchPipeline(),
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
    return ScreenResponse(
        verdict=success.verdict.name,
        phase=success.at_phase.name,
        confidence=success.confidence,
        latency_ms=success.latency_ms,
    )
