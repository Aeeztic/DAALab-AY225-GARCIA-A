"""Request/response models for prediction endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config.settings import BRANCH_ORDER, GENDER_MAP


class PredictRequest(BaseModel):
    """Strict request payload for single-student prediction."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        populate_by_name=True,
    )

    age: int = Field(..., ge=16, le=100)
    cgpa: float = Field(..., ge=0.0, le=10.0)
    backlogs: int = Field(..., ge=0)
    attendance: float = Field(..., ge=0.0, le=100.0)
    tenth_percentage: float = Field(..., ge=0.0, le=100.0)
    twelfth_percentage: float = Field(..., ge=0.0, le=100.0)
    branch: str = Field(..., min_length=1)
    college_tier: int = Field(..., ge=1, le=3)
    python_skill: int = Field(..., ge=0, le=9)
    cpp_skill: int = Field(..., alias="c++_skill", ge=0, le=9)
    java_skill: int = Field(..., ge=0, le=9)
    ml_skill: int = Field(..., ge=0, le=9)
    web_dev_skill: int = Field(..., ge=0, le=9)
    communication_skill: int = Field(..., ge=0, le=9)
    aptitude_score: float = Field(..., ge=0.0, le=100.0)
    logical_reasoning: float = Field(..., ge=0.0, le=100.0)
    internships: int = Field(..., ge=0)
    projects: int = Field(..., ge=0)
    github_projects: int = Field(..., ge=0)
    hackathons: int = Field(..., ge=0)
    certifications: int = Field(..., ge=0)
    coding_contest_rating: float = Field(..., ge=0.0)
    teamwork: int = Field(..., ge=0, le=9)
    leadership: int = Field(..., ge=0, le=9)
    problem_solving: int = Field(..., ge=0, le=9)
    time_management: int = Field(..., ge=0, le=9)
    gender: str = Field(..., min_length=1)
    city_tier: int = Field(..., ge=1, le=3)
    family_income: float = Field(..., ge=0.0)
    include_explanation: bool = False

    @field_validator("branch")
    @classmethod
    def normalize_branch(cls, value: str) -> str:
        branch = value.strip().upper()
        if branch not in BRANCH_ORDER:
            allowed = ", ".join(BRANCH_ORDER)
            raise ValueError(f"branch must be one of: {allowed}")
        return branch

    @field_validator("gender")
    @classmethod
    def normalize_gender(cls, value: str) -> str:
        raw = value.strip()
        mapping = {option.casefold(): option for option in GENDER_MAP}
        canonical = mapping.get(raw.casefold())
        if canonical is None:
            allowed = ", ".join(sorted(GENDER_MAP.keys()))
            raise ValueError(f"gender must be one of: {allowed}")
        return canonical

    def to_model_input(self) -> dict[str, Any]:
        """Return payload shape expected by src.ml.predict.predict_single."""
        return self.model_dump(by_alias=True, exclude={"include_explanation"})


class PredictResponse(BaseModel):
    """Response payload for single-student prediction."""

    model_config = ConfigDict(extra="forbid")

    probability: float = Field(..., ge=0.0, le=1.0)
    prediction: Literal[0, 1]
    shap_explanation: dict[str, float] | None = None
