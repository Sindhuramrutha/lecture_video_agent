"""Assembles the LangGraph pipeline.

The graph is compiled with a checkpointer and interrupts after every
stage, so the caller (main.ipynb) can inspect state and require human
approval before resuming. This is the full pipeline through a final
stitched video with subtitles.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agents.audio_generator import audio_node
from agents.compose import compose_node
from agents.detailed_scripter import detailed_scripter_node
from agents.finalize import finalize_node
from agents.planner import plan_node
from agents.scene_script import scene_script_node
from agents.visual_render import visual_render_node
from state import VideoState

APPROVAL_GATES = [
    "planner",
    "detailed_scripter",
    "scene_script",
    "audio_generator",
    "visual_render",
    "compose",
    "finalize",
]


def build_graph():
    graph = StateGraph(VideoState)
    graph.add_node("planner", plan_node)
    graph.add_node("detailed_scripter", detailed_scripter_node)
    graph.add_node("scene_script", scene_script_node)
    graph.add_node("audio_generator", audio_node)
    graph.add_node("visual_render", visual_render_node)
    graph.add_node("compose", compose_node)
    graph.add_node("finalize", finalize_node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "detailed_scripter")
    graph.add_edge("detailed_scripter", "scene_script")
    graph.add_edge("scene_script", "audio_generator")
    graph.add_edge("audio_generator", "visual_render")
    graph.add_edge("visual_render", "compose")
    graph.add_edge("compose", "finalize")
    graph.add_edge("finalize", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer, interrupt_after=APPROVAL_GATES)
