# Web Demo — Deep-Q Snake (browser-viewable)

A live, browser-viewable version of this project. Visitors click **Watch AI
Play** and the trained `model/model.pth` network drives the snake in real time.

The original project renders with **pygame**, which opens a desktop window and
needs a display — so it can't run on a headless web host as-is. This demo keeps
the exact game rules, 11-value state encoding, and trained model, but:

- reimplements the game loop **headlessly** (no pygame window), and
- renders each frame with **Pillow**, streamed to the browser via **Gradio**.

The result is visually identical (dark checkerboard board, green snake, pulsing
red food, score/gen/length HUD) but viewable by anyone over a URL.

## Contents
| File | Purpose |
|------|---------|
| `app.py` | Gradio UI; streams AI gameplay frames to the browser |
| `snake_game.py` | Headless Snake simulation + state encoding (port of `game.py` / `agent.py` logic) |
| `renderer.py` | Pillow renderer matching the original dark theme |
| `model.py` | `Linear_QNet` architecture (inference only) |
| `model/model.pth` | Trained weights (copied from the repo root `model/`) |
| `requirements.txt` | CPU-only torch, numpy<2, pillow, gradio |
| `DEPLOY.md` | Step-by-step deployment guide |
| `FREE_TIER.md` | Free-tier hosting options and their limits (read first) |
| `render.yaml` | Render Blueprint for one-click free-tier deploy |
| `README_SPACE.md` | README with the HF Spaces YAML header (only if using HF) |

## Run locally
```bash
cd web-demo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py            # open http://127.0.0.1:7860
```

## Deploy (free)
See [`FREE_TIER.md`](./FREE_TIER.md) for the options and their limits, and
[`DEPLOY.md`](./DEPLOY.md) for step-by-step instructions.

**Short version:** Hugging Face Spaces is **no longer free** for a Gradio app on
a personal account (it now requires PRO). The genuinely free paths are:
- **Render** free web service → permanent URL, but sleeps after 15 min idle
  (~1 min cold start). Recommended.
- **Local + `launch(share=True)`** → instant temporary `*.gradio.live` URL
  (~72h, your machine must stay on).

## Notes
- The original `game.py`, `agent.py`, `model.py`, `helper.py` at the repo root
  are **unchanged**. This folder is purely additive.
- `model.pth` here is a copy of the repo's trained weights so the Space is
  self-contained.
