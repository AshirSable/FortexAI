from enum import Enum

from pydantic import BaseModel, Field


class VerdictLabel(str, Enum):
    MALICIOUS = "malicious"
    BENIGN = "benign"
    UNCERTAIN = "uncertain"


class Verdict(BaseModel):
    """Structured output the LLM judge must return for a single classification."""

    verdict: VerdictLabel
    confidence: float = Field(ge=0.0, le=1.0, description="Model's confidence in the verdict, 0-1.")
    reason: str = Field(min_length=1, description="Short justification for the verdict.")
