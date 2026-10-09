# Free-tier hosting — options and limits for this demo

This demo is a **Gradio** app that runs the trained PyTorch model on **CPU**
(the model is ~17 KB and tiny — no GPU needed). Below are the realistic free
options and their real limits, verified against each provider's official docs
(as of October 2026). Provider policies change, so always re-check before you
rely on them.

> ⚠️ **Important correction about Hugging Face Spaces.** Hugging Face recently
> changed its policy. **Creating a Gradio or Docker Space that runs on compute
> now requires a paid plan (PRO, ~$9/month) for personal accounts.** Only
> **static HTML Spaces are free for everyone.** Free personal accounts can
> still host **up to 2 Gradio Spaces, but only on ZeroGPU** (a GPU tier).
> Source: https://huggingface.co/docs/hub/en/spaces-overview and
> https://huggingface.co/docs/hub/en/spaces-gpus
>
> In other words, the old "deploy a Gradio Space for free on CPU Basic" path is
> no longer available to free personal accounts. The options below reflect that.

---

## Option 1 — Render free web service  ✅ recommended free path

[Render](https://render.com) runs a Gradio app for free as a Python **Web
Service**. This is the most straightforward genuinely-free route today.

**Free tier specs & limits** (source: https://render.com/docs/free):
| Item | Limit |
|------|-------|
| Instance | **0.1 CPU / 512 MB RAM** (web service `free` plan) |
| Monthly runtime | **750 free instance hours per workspace / month** |
| Idle behaviour | **Spins down after 15 min with no traffic**; cold start ~1 min on the next request |
| Filesystem | Ephemeral — resets on every redeploy/restart/spin-down (fine here; the model is in the repo) |
| Cost | $0 (not for production use) |

Specs verified against Render's official docs:
https://render.com/docs/free and https://render.com/docs/compute-plans
(web service `free` row = 0.1 CPU / 512 MB).

**Does this project fit?** Yes, with one caveat. PyTorch CPU + Gradio runs within
512 MB for a model this small, and inference is one tiny forward pass per frame.
The caveats are: (1) the free plan is only **0.1 CPU**, so the cold-start import
of PyTorch after an idle period can make the ~1-minute wake feel sluggish —
once running, gameplay is fine; (2) a woken service stays up only while it keeps
getting traffic, and 750 hours/month is enough to keep one service effectively
always-on.

Deploy outline (full steps in `DEPLOY.md`):
1. Push this repo (or the `web-demo/` folder) to your own GitHub.
2. On Render: **New → Web Service**, connect the repo, choose the **Free** plan.
3. Build command: `pip install -r requirements.txt`
   Start command: `python app.py` (bind to `0.0.0.0` and the `$PORT` env var — see note in `DEPLOY.md`).

---

## Option 2 — Hugging Face Space (needs PRO, ~$9/mo — NOT free)

Still the nicest ML-demo experience, but **no longer free for a Gradio Space on
a personal account**. If you have (or get) **HF PRO**, the **CPU Basic**
hardware itself is $0/hr (2 vCPU, 16 GB RAM, 50 GB disk) — you're paying for the
PRO plan, not the hardware. All the files in this folder are already set up for
it (`README_SPACE.md` has the required config header).

Free sub-case: a **static HTML Space** is free, but this demo is a live Python
app, so static hosting can't run the model. Not applicable here.

---

## Option 3 — Run locally + Gradio share link  ✅ free, temporary

For a quick free demo with **zero deployment**, run it on your own machine and
let Gradio create a temporary public URL:

```python
# in app.py, change the last line to:
demo.queue().launch(share=True)
```
Gradio prints a public `https://xxxx.gradio.live` link that **anyone can open
for ~72 hours**, while your machine stays on. Great for showing someone right
now; not a permanent host. Source: https://www.gradio.app/guides/sharing-your-app

---

## Option 4 — Hugging Face ZeroGPU Space (free, GPU, 2 per account)

Free personal accounts may host **up to 2 Gradio Spaces on ZeroGPU**. ZeroGPU is
a shared-GPU tier — overkill for this CPU-only model and it needs small code
decorators (`@spaces.GPU`), so it's not worth the complexity here, but it is a
free HF path if you specifically want to stay on Spaces.

---

## Quick recommendation

| You want… | Use |
|-----------|-----|
| A permanent free URL | **Render free web service** (Option 1) — accept the ~1 min cold start |
| To show someone in the next 5 minutes, free | **Local + `share=True`** (Option 3) |
| The polished HF Spaces experience, willing to pay ~$9/mo | **HF PRO + CPU Basic** (Option 2) |

## Resource footprint of this project
- Model: `Linear_QNet(11, 256, 3)`, weights ~17 KB.
- CPU-only inference; one forward pass per game step — negligible compute.
- Memory: dominated by the PyTorch import (~hundreds of MB), fits Render's 512 MB.
- No database, no persistent storage, no GPU required.
