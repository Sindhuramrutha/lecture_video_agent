"""Assembles the LangGraph pipeline.

The graph is compiled with a checkpointer and interrupts after every
stage, so the caller (main.ipynb) can inspect state and require human
approval before resuming.

Scripting is a single stage: the caller supplies `detailed_script`
directly (already-written textbook-style content — see
utils.references for loading .txt/.md/.pdf/.docx) and scene_script
splits it straight into scenes. There's no planner/detailed_scripter
stage generating that content from a bare topic — add one back only if
a from-a-topic-alone workflow is needed again.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agents.audio_generator import audio_node
from agents.compose import compose_node
from agents.finalize import finalize_node
from agents.scene_script import scene_script_node
from agents.visual_render import visual_render_node
from state import VideoState

APPROVAL_GATES = [
    "scene_script",
    "audio_generator",
    "visual_render",
    "compose",
    "finalize",
]


def build_graph():
    graph = StateGraph(VideoState)
    graph.add_node("scene_script", scene_script_node)
    graph.add_node("audio_generator", audio_node)
    graph.add_node("visual_render", visual_render_node)
    graph.add_node("compose", compose_node)
    graph.add_node("finalize", finalize_node)
    graph.set_entry_point("scene_script")
    graph.add_edge("scene_script", "audio_generator")
    graph.add_edge("audio_generator", "visual_render")
    graph.add_edge("visual_render", "compose")
    graph.add_edge("compose", "finalize")
    graph.add_edge("finalize", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer, interrupt_after=APPROVAL_GATES)
