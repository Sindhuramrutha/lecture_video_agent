"""Local desktop notification so you don't have to watch the notebook
while a stage runs. macOS only (osascript) — fails silently elsewhere
since this is a convenience, not a critical path.
"""

import subprocess


def notify(message: str, title: str = "NeuroGati Video Engine") -> None:
    safe_message = message.replace('"', "'")
    safe_title = title.replace('"', "'")
    try:
        subprocess.run(
            ["osascript", "-e", f'display notification "{safe_message}" with title "{safe_title}"'],
            capture_output=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
