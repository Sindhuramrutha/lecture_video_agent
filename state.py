"""The single state object that flows through the LangGraph pipeline.

Fields are added incrementally as each stage of the pipeline is
built. Do not add fields here for stages that don't exist yet.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class WordTiming(BaseModel):
    word: str
    start: float
    end: float


class Scene(BaseModel):
    id: int
    subtopic: str
    narration: str
    animation_description: str
    scene_type: Literal["manim", "matplotlib", "image_only", "audio_only"]

    # --- produced by the Audio Generator stage ---
    audio_path: Optional[str] = None
    timestamps: list[WordTiming] = Field(default_factory=list)

    # --- produced by the Visual branch (manim scenes only, for now) ---
    render_prompt: Optional[str] = None
    code: Optional[str] = None
    video_path: Optional[str] = None
    render_attempts: int = 0
    render_status: Literal["not_applicable", "rendered", "failed"] = "not_applicable"
    last_feedback: Optional[str] = None

    # --- produced by the Compose stage ---
    composed_path: Optional[str] = None


class VideoState(BaseModel):
    # --- input ---
    topic: str
    audience: str
    duration_minutes: int
    reference_paths: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)

    # --- produced by the Planner agent (unused by the default graph — see workflow.py) ---
    objectives: list[str] = Field(default_factory=list)
    teaching_plan: Optional[str] = None

    # --- input to the Audio-Visual Scene Script agent: already-written textbook-style
    # content, supplied directly by the caller (see utils.references) ---
    detailed_script: Optional[str] = None

    # --- produced by the Audio-Visual Scene Script agent ---
    scenes: list[Scene] = Field(default_factory=list)

    # --- produced by the Finalize stage ---
    subtitles_path: Optional[str] = None
    final_video_path: Optional[str] = None
