# Deploy the Deep-Q Snake demo to Hugging Face Spaces (free)

The demo lives in this `web-demo/` folder and is a self-contained Gradio app:

```
web-demo/
├── app.py            # Gradio UI, streams AI gameplay
├── snake_game.py     # headless Snake simulation + state encoding
├── renderer.py       # Pillow frame renderer (original dark theme)
├── model.py          # Linear_QNet architecture (inference only)
├── model/model.pth   # trained weights
├── requirements.txt
├── README.md         # this folder's readme (repo view)
└── README_SPACE.md   # README WITH the HF Spaces YAML header — use this on the Space
```

Hugging Face Spaces free tier (CPU Basic: 2 vCPU / 16 GB RAM) is plenty — the
model is tiny (~17 KB) and runs on CPU.

> Important: a Hugging Face Space needs a `README.md` whose first lines are the
> YAML config header. In this repo that header lives in `README_SPACE.md` (to
> avoid clashing with the folder's own readme). **When you deploy, rename
> `README_SPACE.md` to `README.md` on the Space.**

---

## Option A — Deploy from the website (easiest, no CLI)

1. Create a free account at https://huggingface.co/join.
2. Go to https://huggingface.co/new-space.
3. Fill in:
   - **Owner:** your username
   - **Space name:** e.g. `deep-q-snake`
   - **License:** MIT
   - **SDK:** select **Gradio**
   - **Hardware:** **CPU basic — Free**
   - **Visibility:** Public
4. Click **Create Space**.
5. On the new Space page, open the **Files** tab → **Add file** → **Upload files**,
   and upload the contents of this `web-demo/` folder, keeping the
   `model/model.pth` path (create the `model` folder by typing `model/model.pth`
   as the path when uploading). Upload: `app.py`, `snake_game.py`, `renderer.py`,
   `model.py`, `requirements.txt`, `model/model.pth`, and `README_SPACE.md`
   **renamed to `README.md`**.
6. The Space builds automatically. After a few minutes the status turns
   **Running** and your public URL is:
   `https://huggingface.co/spaces/<your-username>/deep-q-snake`
7. Open it, click **▶ Watch AI Play**. Done — share that URL with anyone.

---

## Option B — Deploy with Git (recommended for updates)

Requires: a free HF account and a Hugging Face access token
(https://huggingface.co/settings/tokens — create one with **write** access).

```bash
# 1. Install the HF CLI and log in
pip install -U huggingface_hub
huggingface-cli login        # paste your write token

# 2. Create the Space (one time)
huggingface-cli repo create deep-q-snake --type space --space_sdk gradio

# 3. Clone the (empty) Space repo somewhere separate
git clone https://huggingface.co/spaces/<your-username>/deep-q-snake
cd deep-q-snake

# 4. Copy the demo files in (run from the root of this repo clone)
cp -r /path/to/Deep-Q-Snake/web-demo/{app.py,snake_game.py,renderer.py,model.py,requirements.txt} .
mkdir -p model
cp /path/to/Deep-Q-Snake/web-demo/model/model.pth model/
cp /path/to/Deep-Q-Snake/web-demo/README_SPACE.md README.md   # header README

# 5. Push
git add .
git commit -m "Deep-Q Snake live Gradio demo"
git push
```

The Space rebuilds on every push. Watch the **Logs** tab for build progress;
when it says **Running**, your public URL is live.

> `model.pth` is only ~17 KB so plain Git is fine (no Git LFS needed).

---

## Run locally first (optional sanity check)

```bash
cd web-demo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
# open http://127.0.0.1:7860
```

---

## Troubleshooting

- **Build fails on torch:** the `--extra-index-url .../cpu` line in
  `requirements.txt` pulls the CPU-only wheel, which is smaller and correct for
  the free CPU tier. Keep it.
- **NumPy ABI error:** `numpy<2` is pinned on purpose (torch 2.2.2 needs NumPy 1.x).
- **Snake plays randomly:** means `model/model.pth` wasn't uploaded to the
  `model/` subfolder. Re-check the file path on the Space.
- **Space sleeps after inactivity:** free Spaces pause when idle and wake on the
  next visit (a few seconds to resume). This is normal for the free tier.
