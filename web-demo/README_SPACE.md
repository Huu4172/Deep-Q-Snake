---
title: Deep-Q Snake
emoji: 🐍
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
---

# 🐍 Deep-Q Snake — Live AI Demo

A browser-viewable demo of a Deep Q-Learning agent (PyTorch) that taught itself
to play Snake. Click **Watch AI Play** and the trained network drives the snake
live in your browser.

This is a web demo of [ish7nw/Deep-Q-Snake](https://github.com/ish7nw/Deep-Q-Snake).
The original project uses pygame (a desktop window); this demo reimplements the
game loop headlessly and renders frames with Pillow so it runs on a server and
streams to the browser.

## How it works
- **Network:** `Linear_QNet`, 11 inputs → 256 hidden (ReLU) → 3 outputs (one Q-value per move).
- **State (11 inputs):** danger straight / right / left · heading (4) · food direction (4).
- **Actions:** go straight, turn right, turn left.
- **Inference:** greedy — the move with the highest Q-value is chosen each step (no exploration).

## Files
| File | Purpose |
|------|---------|
| `app.py` | Gradio UI that streams AI gameplay frames |
| `snake_game.py` | Headless Snake simulation + 11-value state encoding (port of the original `game.py` / `agent.py`) |
| `renderer.py` | Pillow renderer matching the original dark theme |
| `model.py` | `Linear_QNet` architecture (inference only) |
| `model/model.pth` | Trained weights |

## Run locally
```bash
pip install -r requirements.txt
python app.py
```
Then open the printed local URL (default http://127.0.0.1:7860).

## Credits
Model and original game by [ish7nw](https://github.com/ish7nw/Deep-Q-Snake).
