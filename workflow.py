"""Assembles the LangGraph pipeline.

The graph is compiled with a checkpointer and interrupts after every
agent, so the caller (main.py) can inspect state and require human
approval before resuming. Only the Planner node exists so far.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agents.planner import plan_node
from state import VideoState

APPROVAL_GATES = ["planner"]


def build_graph():
    graph = StateGraph(VideoState)
    graph.add_node("planner", plan_node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer, interrupt_after=APPROVAL_GATES)
