from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .analysis import analyze_requirements, build_traceability

app = FastAPI(
    title="AgentReq",
    version="0.1.0",
    description="Trustworthy AI-assisted requirements engineering research prototype."
)

WEB = Path(__file__).resolve().parent.parent / "web" / "index.html"


class Requirement(BaseModel):
    id: str
    text: str = Field(min_length=5)


class Artifact(BaseModel):
    id: str
    type: str
    text: str = Field(min_length=5)


class AnalyzeRequest(BaseModel):
    requirements: list[Requirement]


class TraceRequest(BaseModel):
    requirements: list[Requirement]
    artifacts: list[Artifact]
    threshold: float = Field(default=0.18, ge=0, le=1)


class ReviewRequest(BaseModel):
    requirement_id: str
    artifact_id: str
    decision: str


@app.get("/health")
def health():
    return {"status": "ok", "service": "agentreq", "version": "0.1.0"}


@app.get("/")
def index():
    return FileResponse(WEB)


@app.post("/analyze")
def analyze(payload: AnalyzeRequest):
    return {"results": analyze_requirements([r.model_dump() for r in payload.requirements])}


@app.post("/trace")
def trace(payload: TraceRequest):
    return {
        "candidates": build_traceability(
            [r.model_dump() for r in payload.requirements],
            [a.model_dump() for a in payload.artifacts],
            payload.threshold,
        )
    }


@app.post("/review")
def review(payload: ReviewRequest):
    decision = payload.decision.lower()
    if decision not in {"accept", "reject"}:
        return {"error": "decision must be accept or reject"}
    return {
        "requirement_id": payload.requirement_id,
        "artifact_id": payload.artifact_id,
        "decision": decision,
        "status": "recorded",
    }
