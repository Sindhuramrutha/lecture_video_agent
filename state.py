"""The single state object that flows through the LangGraph pipeline.

Fields are added incrementally as each stage of the pipeline is
built. Do not add fields here for stages that don't exist yet.
"""

from typing import Optional

from pydantic import BaseModel, Field


class VideoState(BaseModel):
    # --- input ---
    topic: str
    audience: str
    duration_minutes: int

    # --- produced by the Planner agent ---
    objectives: list[str] = Field(default_factory=list)
    teaching_plan: Optional[str] = None

    # --- produced by later agents (not implemented yet) ---
    script: Optional[str] = None
    storyboard: Optional[str] = None
