"""Shared exception for a renderer-agnostic render failure.

Both utils/manim_render.py and utils/matplotlib_render.py raise this same
class so agents/visual_render.py's shared retry loop can catch either
renderer's failures with one except clause, regardless of which renderer
a scene was routed to.
"""


class RenderError(Exception):
    pass
