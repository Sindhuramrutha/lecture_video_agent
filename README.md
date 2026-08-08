# NeuroGati Video Engine

Internal video generation automation pipeline for Neurovidya by Neurogati.
Turns written content (a topic, or a document you author) into a narrated,
animated video: script → narration audio → Manim/Matplotlib animation →
composed scene clips → a stitched final video with subtitles.

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
cd video_engine
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
| `CODE_LLM_BASE_URL`, `CODE_LLM_API_KEY`, `CODE_LLM_MODEL` | Optional separate model for Manim code generation; falls back to the `LLM_*` values above if unset |
| `TTS_VOICE` | Kokoro TTS voice (default `af_heart`) |

Never commit `.env` — it's already git-ignored.

## 5. Create a project

`projects/` ships empty (only `.gitkeep`) — every project folder is created
locally by you or by the tooling, never pulled from git, since it holds
generated media that's specific to each run.

Two ways to start:

### Option A — from your own written content (recommended, staged CLI)

1. Write your content by hand as textbook-style prose (sections, worked
   examples, explanations — not bullet notes). See
   [CONTENT_GUIDELINES.md](CONTENT_GUIDELINES.md) for the full authoring
   rules — this is the one input you write; everything downstream is
   generated from it.
2. Create the input folder and drop your file in:

   ```bash
   mkdir -p projects/my-project/input_files
   cp ~/ch1.docx projects/my-project/input_files/
   ```

   Supported formats: `.txt`, `.md`, `.pdf`, `.docx`.

### Option B — from a bare topic (notebook, interactive)

Open `main.ipynb`, edit the `topic` / `audience` / `duration_minutes` cell,
and run cells one stage at a time — it pauses after each stage
(script → audio → visual render → compose → finalize) so you can inspect and
approve before continuing. This drives an LLM to invent the content itself;
prefer Option A if you want to control exactly what's said.

## 6. Render the video

Using the staged CLI (Option A projects), from the `video_engine` directory
with the conda env active:

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

Other useful invocations:

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
```

Flags: `--project` (folder under `projects/`), `--input` (filename under
`projects/<project>/input_files/`), `--topic` (defaults to the input
filename), `--audience` (default: "students new to the topic"),
`--duration-minutes` (default 15).

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

## 7. Output layout

Everything below is generated under `projects/<project>/`, per chapter slug
(derived from the input filename), and stays local (git-ignored):

```
projects/<project>/
  input_files/<input>                -- your input (you provide this)
  <chapter-slug>/
    state.json                       -- scene state, resumed across invocations
    scene-wise-script/               -- narration + animation description per scene
    prompts/                         -- timed animation briefs
    audio/                           -- narration audio (Kokoro + word alignment)
    manim_code/                      -- generated Manim source
    manim/                           -- raw Manim render working dirs
    rendered_video/                  -- scene_NN_silent.mp4 (review) and
                                         scene_NN_with_audio.mp4 (after compose)
  videos/<chapter-slug>.mp4          -- final stitched video
  videos/<chapter-slug>.srt          -- subtitles
```

## Repo layout (tracked in git)

```
agents/      -- pipeline stages (script, audio, prompts, code, render, compose)
utils/       -- shared helpers (ffmpeg, Manim rendering/linting, TTS, alignment, storage)
config/      -- environment/settings loading
state.py     -- VideoState / Scene models shared across the pipeline
workflow.py  -- LangGraph graph wiring for the notebook-driven workflow
main.ipynb   -- interactive, topic-driven orchestrator (Option B above)
run_chapter.py, run_project.py  -- staged CLI drivers (Option A above)
projects/    -- empty in git (.gitkeep only); your generated projects live here locally
```
