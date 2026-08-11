"""Static pre-render checks for generated Manim code.

Catches known bug classes -- ImageMobject/SVGMobject on a file that isn't a
real registered asset, MathTex/Tex needing an absent LaTeX install,
self.camera.frame on a plain Scene -- with a cheap regex pass before ever
invoking Manim. Every one of these has actually crashed a real render in
this pipeline; catching it here costs milliseconds, versus a render attempt
(and the LLM call that produced it) burned on a mistake that was already
known to be fatal.
"""

import re

from utils.asset_library import image_asset_paths

_IMAGE_CTOR = re.compile(r"\b(?:ImageMobject|SVGMobject)\s*\(")
_ASSET_PATHS_ARG = re.compile(r"\s*ASSET_PATHS\[\s*['\"]([^'\"]+)['\"]\s*\]")
_STRING_LITERAL_ARG = re.compile(r"\s*['\"]([^'\"]+)['\"]")
_SAVEFIG_TARGET = re.compile(r"\.savefig\(\s*['\"]([^'\"]+)['\"]")
_FORBIDDEN_LATEX = re.compile(r"\b(MathTex|Tex)\s*\(")
_CAMERA_FRAME = re.compile(r"self\.camera\.frame\b")
_CLASS_DECL = re.compile(r"class\s+(\w+)\s*\(([^)]*)\)\s*:")
_STRAY_FENCE = re.compile(r"^```", re.MULTILINE)
_CODE_CTOR = re.compile(r"(?:(\w+)\s*=\s*)?Code\(")
_FIT_CHAINED = re.compile(r"^\s*\.(scale_to_fit_width|scale_to_fit_height|scale)\(")


def _find_matching_paren(text: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def lint_code(code: str, class_name: str) -> list[str]:
    """Returns a list of problem descriptions; empty means the code passed."""
    issues = []

    # Checked first, and returned alone if it fails: a SyntaxError means the code
    # never even parses, so none of the pattern-based checks below are meaningful
    # (they'd just add confusing, possibly-wrong noise on top of the real problem).
    # This catches this class of failure generically -- e.g. a response that got
    # cut off mid-expression by hitting the token limit on a long/complex scene,
    # or mismatched parens from any cause -- rather than needing a hand-written
    # rule for every specific way generated code can fail to parse.
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError as exc:
        lines = code.split("\n")
        lineno = exc.lineno or len(lines)
        context_start = max(0, lineno - 4)
        context = "\n".join(
            f"{'>>> ' if i + 1 == lineno else '    '}{i + 1}: {lines[i]}"
            for i in range(context_start, min(len(lines), lineno))
        )
        near_end = lineno >= len(lines) - 3
        return [
            f"Code does not parse as valid Python: {exc.msg} at line {exc.lineno}.\n"
            f"Context (>>> marks the problem line):\n{context}\n"
            + (
                "This is near the very end of the response -- the code was likely cut off "
                "before finishing. Write the scene more concisely (fewer, larger beats; less "
                "repeated/boilerplate code; use a loop instead of writing out each repeated "
                "element by hand) so it fits completely within the response."
                if near_end
                else "Fix the syntax error shown above."
            )
        ]

    if _STRAY_FENCE.search(code):
        issues.append(
            "Contains a ``` markdown fence line -- this always causes a SyntaxError. "
            "Respond with only the raw Python source, no fences anywhere in the response."
        )
    valid_assets = image_asset_paths()
    savefig_targets = set(_SAVEFIG_TARGET.findall(code))
    for ctor_match in _IMAGE_CTOR.finditer(code):
        arg_match = _ASSET_PATHS_ARG.match(code, ctor_match.end())
        if arg_match is not None:
            key = arg_match.group(1)
            if key not in valid_assets:
                valid = ", ".join(repr(k) for k in valid_assets) or "(none registered)"
                issues.append(
                    f"ASSET_PATHS[{key!r}] is not a registered asset. Valid keys: {valid}. Use "
                    f"one of those, or build the visual from Manim's own vector primitives instead."
                )
            continue
        literal_match = _STRING_LITERAL_ARG.match(code, ctor_match.end())
        if literal_match is not None and literal_match.group(1) in savefig_targets:
            continue  # a locally-generated matplotlib plot, produced by savefig() in this same file
        issues.append(
            "Uses ImageMobject or SVGMobject with a path that's neither an ASSET_PATHS[...] "
            "lookup nor a filename this same file produced via matplotlib's savefig() -- any "
            "other path is not a real file and always crashes. Only reference a key from the "
            "available assets list in the system prompt (e.g. SVGMobject(ASSET_PATHS[\"heart\"])), "
            "load a filename you just saved yourself with savefig(), or build the visual from "
            "Manim's own vector primitives instead."
        )
    if _FORBIDDEN_LATEX.search(code):
        issues.append(
            "Uses MathTex or Tex -- both shell out to a LaTeX install that isn't "
            "present in this environment. Use plain Text with unicode instead."
        )

    for ctor_match in _CODE_CTOR.finditer(code):
        var_name = ctor_match.group(1)
        open_paren = ctor_match.end() - 1
        close_paren = _find_matching_paren(code, open_paren)
        if close_paren == -1:
            continue
        after = code[close_paren + 1 :]
        if _FIT_CHAINED.match(after):
            continue  # chained directly onto the constructor, e.g. Code(...).scale_to_fit_width(10)
        if var_name and re.search(rf"\b{re.escape(var_name)}\.(scale_to_fit_width|scale_to_fit_height|scale)\(", after):
            continue  # fitted on a later line via the variable it was assigned to
        issues.append(
            "A Code(...) block isn't followed by a size cap (.scale_to_fit_width(...) / "
            ".scale_to_fit_height(...) / .scale(...)) -- Code renders at whatever size its "
            "text naturally comes out to, which routinely overflows the ~14-unit-wide frame "
            "for anything but a single short line. Add .scale_to_fit_width(10) (or "
            ".scale_to_fit_height(5) for many lines) chained onto the Code(...) call or on the "
            "variable it's assigned to, before positioning it."
        )

    class_match = _CLASS_DECL.search(code)
    if class_match is None:
        issues.append(f"No `class {class_name}(...):` definition found.")
    else:
        found_name, bases = class_match.group(1), class_match.group(2)
        if found_name != class_name:
            issues.append(f"Class is named '{found_name}', must be named exactly '{class_name}'.")
        if _CAMERA_FRAME.search(code) and "MovingCameraScene" not in bases:
            issues.append(
                "Uses self.camera.frame but the class doesn't subclass MovingCameraScene -- "
                "that attribute only exists there, plain Scene raises AttributeError. Animate "
                "the mobjects themselves (.animate.scale()/.move_to()) instead of the camera."
            )

    return issues
