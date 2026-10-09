# Deploy the Deep-Q Snake demo

The demo lives in this `web-demo/` folder and is a self-contained Gradio app:

```
web-demo/
├── app.py            # Gradio UI, streams AI gameplay (binds 0.0.0.0:$PORT)
├── snake_game.py     # headless Snake simulation + state encoding
├── renderer.py       # Pillow frame renderer (original dark theme)
├── model.py          # Linear_QNet architecture (inference only)
├── model/model.pth   # trained weights
├── requirements.txt
├── render.yaml       # Render Blueprint (free-tier web service)
├── README.md         # this folder's readme
├── README_SPACE.md   # README WITH the HF Spaces YAML header (if deploying to HF)
├── FREE_TIER.md      # free-tier options and their limits (read this first)
└── DEPLOY.md         # this file
```

> 💡 **Read [`FREE_TIER.md`](./FREE_TIER.md) first.** It explains which options
> are genuinely free and their limits. In short: **Hugging Face Spaces is no
> longer free for a Gradio app on a personal account** (it now needs PRO). The
> free paths are **Render** (permanent URL, cold starts) or **local +
> `share=True`** (temporary URL).

---

## Option 1 — Render free web service  ✅ recommended free path

Free tier: 512 MB RAM, 750 instance-hours/month, **spins down after 15 min idle**
(≈1 min cold start on the next visit). Enough for this tiny CPU model.

### A) One-click via the included Blueprint
1. Push this repo to **your own GitHub** (fork or your copy).
2. Go to https://render.com → sign up (free) → **New → Blueprint**.
3. Connect the repo. Render reads `web-demo/render.yaml` and provisions a free
   web service automatically.
4. When the build finishes, your public URL is
   `https://deep-q-snake-XXXX.onrender.com`.

### B) Manual (no Blueprint)
1. Push the repo to your GitHub.
2. Render → **New → Web Service** → connect the repo.
3. Settings:
   - **Root Directory:** `web-demo`
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python app.py`
   - **Instance Type:** **Free**
4. Create → wait for the build → open the generated URL.

`app.py` already binds `0.0.0.0` and the `$PORT` env var Render provides, so no
extra config is needed.

---

## Option 2 — Local + temporary public link  ✅ free, instant, no deploy

Fastest way to show someone right now (your machine must stay on; link lasts ~72h):

```bash
cd web-demo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# then launch with a share link:
python -c "import app; app.demo.queue().launch(share=True)"
```
Gradio prints a public `https://xxxx.gradio.live` URL.

---

## Option 3 — Hugging Face Space (requires PRO, ~$9/mo — not free)

Only do this if you have **HF PRO** (a Gradio Space on a free personal account is
no longer allowed; see `FREE_TIER.md`). On PRO, the **CPU Basic** hardware is
$0/hr.

1. https://huggingface.co/new-space → SDK **Gradio**, Hardware **CPU basic**, Public.
2. Upload the contents of this `web-demo/` folder, keeping the `model/model.pth`
   path, and rename `README_SPACE.md` → `README.md` on the Space (it carries the
   required YAML header).
3. The Space builds; when it shows **Running**, the URL is
   `https://huggingface.co/spaces/<username>/deep-q-snake`.

Git-based alternative:
```bash
pip install -U huggingface_hub
huggingface-cli login
huggingface-cli repo create deep-q-snake --type space --space_sdk gradio
git clone https://huggingface.co/spaces/<username>/deep-q-snake && cd deep-q-snake
cp -r /path/to/web-demo/{app.py,snake_game.py,renderer.py,model.py,requirements.txt} .
mkdir -p model && cp /path/to/web-demo/model/model.pth model/
cp /path/to/web-demo/README_SPACE.md README.md
git add . && git commit -m "Deep-Q Snake Gradio demo" && git push
```

---

## Run locally (sanity check, no public link)

```bash
cd web-demo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py          # http://127.0.0.1:7860
```

---

## Troubleshooting

- **Build fails on torch:** the `--extra-index-url .../cpu` line in
  `requirements.txt` pulls the CPU-only wheel (smaller, correct for free CPU tiers).
- **NumPy ABI error:** `numpy<2` is pinned on purpose (torch 2.2.2 needs NumPy 1.x).
- **Snake plays randomly:** `model/model.pth` wasn't deployed to the `model/`
  subfolder — check the path.
- **Render: first visit is slow (~1 min):** expected — the free service was
  asleep and is waking up. Subsequent requests are fast.
- **512 MB memory on Render:** fine for this model. If you ever hit a limit from
  heavier torch builds, the CPU-only wheel (already pinned) keeps the footprint down.
