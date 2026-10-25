# Lecture Video Agent

An agentic LLM pipeline that turns written content you author into a narrated, animated video: script →
narration audio → Manim/Matplotlib animation → composed scene clips → a
stitched final video with subtitles.

## 1. Prerequisites

- Linux/macOS/Windows, Python 3.12
- [Conda](https://docs.conda.io/) (Miniconda is enough) — see install steps below
- `ffmpeg` — required for audio/video muxing (not available on PyPI); see the
  OS-specific instructions in step 3 — **Windows needs a different install path
  than Linux/macOS**, see the note there
- An LLM API key (DeepSeek by default, or your own OpenAI-compatible endpoint
  such as a self-hosted vLLM server)

A LaTeX distribution is **not** required — the code generator is instructed to
never use `MathTex`/`Tex` so this pipeline never needs one.

## 2. Install Conda

Skip this if `conda` is already available (`conda --version`).

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda init bash   # or: conda init zsh
```

Restart your shell (or `source ~/.bashrc`) so the `conda` command is on `PATH`.

## 3. Create the environment

```bash
cd lecture-video-agent
conda create -n video_engine python=3.12 -y
conda activate video_engine

# Python dependencies:
pip install -r requirements.txt
```

`requirements.txt` pins `manim==0.20.1` exactly — the code generator and
`utils/manim_reference.py` are written against that version's API, and a
different Manim version can rename/remove attributes this pipeline depends on.

### Installing ffmpeg

**Linux/macOS:**

```bash
conda install -c conda-forge ffmpeg -y
```

**Windows: do not use `conda install -c conda-forge ffmpeg`.** That package is a
shared-library build (`ffprobe.exe` depends on separately-versioned DLLs sitting
next to it), and on some Windows machines it installs an `ffprobe.exe` that
crashes immediately with `STATUS_ENTRYPOINT_NOT_FOUND` (a Python traceback
ending in `subprocess.CalledProcessError: ... exit status 3221225785`) due to a
mismatched/corrupted DLL. Use a static build instead — it's self-contained, so
this failure mode can't happen:

```powershell
winget install Gyan.FFmpeg
```

Then open a **new** terminal (so `PATH` picks up the change), reactivate the
conda env, and verify:

```powershell
conda activate video_engine
ffprobe -version
```

This should print version info. If it doesn't, or `ffprobe` isn't found at all,
see the "ffmpeg preflight check" note below — `run_chapter.py`/`run_project.py`
will also catch and explain this automatically the moment you try to run them.

### Verifying the install

Both entry-point scripts (`run_chapter.py`, `run_project.py`) check that
`ffmpeg`/`ffprobe` actually run before doing any work, and exit immediately
with an actionable message if not — rather than failing several minutes into a
run, after the script/audio stages have already completed. You can also check
manually any time with:

```bash
ffprobe -version
```

## 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in:

| Variable | Purpose |
|---|---|
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | Script/prompt-generation LLM. Defaults to DeepSeek; point at a self-hosted vLLM server instead by setting `LLM_BASE_URL=http://localhost:8800/v1`, `LLM_API_KEY=not-needed`, `LLM_MODEL=<your served model>` |
| `CODE_LLM_BASE_URL`, `CODE_LLM_API_KEY`, `CODE_LLM_MODEL` | Code-generation model, used by both the Manim and Matplotlib coders; falls back to the `LLM_*` values above if unset. Any OpenAI-compatible endpoint works here, e.g. Claude via OpenRouter: `CODE_LLM_BASE_URL=https://openrouter.ai/api/v1`, `CODE_LLM_MODEL=anthropic/claude-sonnet-5` |
| `TTS_VOICE` | Kokoro TTS voice (default `af_heart`) |

Never commit `.env` — it's already git-ignored.

## 5. Create a project

`projects/` ships empty (only `.gitkeep`) — every project folder is created
locally by you or by the tooling, never pulled from git, since it holds
generated media that's specific to each run.

1. Write your content by hand as textbook-style prose (sections, worked
   examples, explanations — not bullet notes). See
   [CONTENT_GUIDELINES.md](CONTENT_GUIDELINES.md) for the full authoring
   rules — this is the one input you write; everything downstream is
   generated from it. If you state a target duration in the file (e.g.
   `Target duration: 9 minutes`), treat it as a target for how much you
   write, not a lever the pipeline can pull — it narrates your content
   faithfully and will not pad or invent material to reach a duration your
   source text doesn't actually contain.
2. Create the input folder and drop your file in:

   ```bash
   mkdir -p projects/my-project/input_files
   cp ~/ch1.docx projects/my-project/input_files/
   ```

   Supported formats: `.txt`, `.md`, `.pdf`, `.docx`.

## 6. Render the video

From the `lecture-video-agent` directory with the conda env active:

```bash
# One-halt workflow (recommended): script -> audio -> prompts -> silent render,
# then stop so you can review before audio gets muxed in
python run_chapter.py --project my-project --input ch1.docx --stage prepare

# Review: watch projects/my-project/<chapter-slug>/rendered_video/scene_*_silent.mp4

# Fix a specific scene if something looks wrong (repeat as needed)
python run_chapter.py --project my-project --input ch1.docx \
    --scenes 2 --revise "The neuron highlight boxes overlap, fade each one out before the next appears"

# Happy with everything -- mux audio in and stitch the final video
python run_chapter.py --project my-project --input ch1.docx --stage compose
```

### `run_chapter.py` flag reference

| Flag | Purpose |
|---|---|
| `--project` | Required. Folder under `projects/`, e.g. `my-project`. |
| `--input` | Required. Filename under `projects/<project>/input_files/`. |
| `--topic` | Defaults to the input filename. |
| `--audience` | Default: `"students new to the topic"`. |
| `--duration-minutes` | Default: `15`. Only matters if your content doesn't state its own target — see section 5; the pipeline never pads to hit either number. |
| `--stage` | One of `all`, `prepare`, `script`, `audio`, `prompts`, `render`, `compose` — see recipes below. |
| `--scenes` | Comma-separated scene ids, e.g. `1,3`. Scopes `render`/`compose` to a subset; required (as a single id) for `--revise`, `--explain`, `--set-scene-type`. |
| `--revise "<feedback>"` | Regenerate one scene's code from your plain-English feedback plus its existing code, then re-render it. Requires `--scenes <single id>`. |
| `--explain` | Ask the code-generation LLM to explain a failed scene's saved error log in plain English, plus suggest `--revise` feedback. Requires `--scenes <single id>`. No separate account needed — reuses `CODE_LLM_*`. |
| `--set-scene-type {manim,matplotlib,image_only,audio_only}` | Override one scene's type — e.g. the classifier picked `image_only` for a scene with real narration, and you want actual animation instead. Requires `--scenes <single id>`. Clears that scene's stale render state (code/video/status) so it starts fresh under the new type; doesn't render by itself — follow with `--stage render --scenes <id>`. |

### Command recipes

```bash
# Run everything straight through, no review pause
python run_chapter.py --project my-project --input ch1.docx --stage all

# Run one stage at a time
python run_chapter.py --project my-project --input ch1.docx --stage script
python run_chapter.py --project my-project --input ch1.docx --stage audio
python run_chapter.py --project my-project --input ch1.docx --stage prompts
python run_chapter.py --project my-project --input ch1.docx --stage render  [--scenes 1,3]
python run_chapter.py --project my-project --input ch1.docx --stage compose [--scenes 1,3]

# Ask the LLM to explain a failed scene's error in plain English, with a
# suggested --revise fix
python run_chapter.py --project my-project --input ch1.docx --scenes 2 --explain

# The classifier picked the wrong scene type for a scene (e.g. image_only for
# a scene with 30s of narration, which just sits on a static title card) --
# override it, then render it for real:
python run_chapter.py --project my-project --input ch1.docx --scenes 2 --set-scene-type manim
python run_chapter.py --project my-project --input ch1.docx --scenes 2 --stage render

# Re-render every scene of a given type after a prompt/model change, without
# touching script/audio/prompts (find the ids first, e.g. via state.json)
python run_chapter.py --project my-project --input ch1.docx --scenes 1,4,7 --stage render
```

### Lower-level: `run_project.py`

For quick iteration when you already have per-scene prompts and audio in
place (`prompts/scene_NN_prompt.txt` + `audio/scene_NN.wav` under
`projects/<project>/`, skipping script/audio generation entirely):

```bash
python run_project.py --project python-course --stage all              # all scenes
python run_project.py --project python-course --stage code --scenes 1,3
python run_project.py --project python-course --stage render --scenes 1
python run_project.py --project python-course --stage compose
```

| Flag | Purpose |
|---|---|
| `--project` | Required. Folder under `projects/`. |
| `--scenes` | Comma-separated scene ids; default is every scene with a `prompts/scene_NN_prompt.txt` file. |
| `--stage` | One of `all`, `code` (lint-only, fast iteration, no render), `render`, `compose`. |

`run_project.py` only ever generates Manim scenes (no scene-type classifier, no matplotlib routing) — it's the narrower sibling of `run_chapter.py` for when you already have briefs and audio in place and just want to iterate on code.

## 7. Output layout

Everything below is generated under `projects/<project>/`, per chapter slug
(derived from the input filename), and stays local (git-ignored):

```
projects/<project>/
  input_files/<input>                -- your input (you provide this)
  <chapter-slug>/
    state.json           -- scene state, resumed across invocations
    scene-wise-script/    -- narration + animation description per scene
    prompts/              -- timed animation briefs
    audio/                -- narration audio (Kokoro + word alignment)
    transcripts/          -- word-level timing per scene
    manim/SceneNN/         -- generated source + render working dir, per manim
                              scene (scene-type classifier decides manim vs.
                              matplotlib per scene, see section 5)
    matplotlib/SceneNN/    -- same, for matplotlib scenes
    rendered_video/        -- scene_NN_silent.mp4 (review) and
                              scene_NN_with_audio.mp4 (after compose)
  videos/<chapter-slug>.mp4          -- final stitched video
  videos/<chapter-slug>.srt          -- subtitles
```

A shared asset library also lives at the repo root, not per-project — see
[utils/asset_library.py](utils/asset_library.py). `assets/library.json` is
the manifest; `assets/image/` holds the actual files. Registered assets are
available to every project's Manim scenes via the `ASSET_PATHS` dict (see
`agents/scene_coder.py`'s system prompt for exactly how).

## Repo layout (tracked in git)

```
agents/      -- pipeline stages: scene_script (classifier), audio_generator,
                prompt_maker, scene_coder (Manim), matplotlib_coder, visual_render
                (dispatches each scene to its renderer), compose, finalize
utils/       -- shared helpers: ffmpeg, Manim/Matplotlib rendering + linting,
                manim_reference.py (live-introspected API reference), asset_library,
                TTS, word alignment, storage
config/      -- environment/settings loading
state.py     -- VideoState / Scene models shared across the pipeline
workflow.py  -- LangGraph graph wiring for the notebook-driven workflow (main.ipynb)
main.ipynb   -- interactive, topic-driven orchestrator; drives an LLM to invent
                content from a bare topic instead of narrating a file you wrote --
                see CONTENT_GUIDELINES.md if you want control over exactly what's said
run_chapter.py, run_project.py  -- staged CLI drivers, see sections 5-6 above
assets/      -- shared asset library (manifest + registered image files)
projects/    -- empty in git (.gitkeep only); your generated projects live here locally
```
