"""Audio Generator stage: synthesizes narration audio and aligns timestamps per scene.

Deterministic tool execution, not an LLM decision — this stage runs
Piper TTS and Whisper alignment for every scene. No agent reasoning
happens here, which is why it's not prompt-driven like the other
stages.
"""

from tqdm.auto import tqdm

from state import VideoState
from utils.audio_alignment import align
from utils.notify import notify
from utils.storage import project_dir
from utils.tts import synthesize


def audio_node(state: VideoState) -> dict:
    audio_dir = project_dir(state.topic) / "audio"
    updated_scenes = []
    for scene in tqdm(state.scenes, desc="Audio Generator"):
        out_path = audio_dir / f"scene_{scene.id:02d}.wav"
        synthesize(scene.narration, out_path)
        timestamps = align(out_path)
        updated_scenes.append(
            scene.model_copy(update={"audio_path": str(out_path), "timestamps": timestamps})
        )
    notify(f"Audio Generator done — {len(updated_scenes)} scene(s)")
    return {"scenes": updated_scenes}
