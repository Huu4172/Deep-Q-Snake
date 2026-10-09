"""
Deep-Q Snake — live browser demo (Gradio).

Loads the trained Linear_QNet weights (model/model.pth) and streams the AI
playing Snake, frame by frame, into the browser. Deployable free on
Hugging Face Spaces.
"""
import time

import numpy as np
import torch
import gradio as gr

from model import Linear_QNet
from snake_game import SnakeGame
from renderer import render

# ---- load the trained agent once at startup ----
model = Linear_QNet(11, 256, 3)
loaded = model.load("model.pth")
if loaded:
    print("Loaded trained weights from model/model.pth")
else:
    print("WARNING: model/model.pth not found — the snake will play with random weights.")
model.eval()

MAX_STEPS = 2000  # safety cap per episode so a session can't run forever


def get_action(state):
    """Greedy action from the trained network (no exploration)."""
    state0 = torch.tensor(state, dtype=torch.float)
    with torch.no_grad():
        prediction = model(state0)
    move = int(torch.argmax(prediction).item())
    final_move = [0, 0, 0]
    final_move[move] = 1
    return final_move


def play(speed):
    """Generator: yields (frame, status) tuples to stream gameplay to the UI."""
    game = SnakeGame()
    best = 0
    steps = 0
    frame_delay = {"Slow": 0.09, "Normal": 0.045, "Fast": 0.015}.get(speed, 0.045)

    while True:
        state = game.get_state()
        action = get_action(state)
        game_over, score = game.play_step(action)
        steps += 1
        best = max(best, score)

        status = f"Episode {game.generation}  ·  Score {score}  ·  Best {best}"
        yield render(game), status

        if game_over:
            time.sleep(0.6)
            game.reset()
            steps = 0
            continue

        if steps >= MAX_STEPS:
            game.reset()
            steps = 0

        time.sleep(frame_delay)


with gr.Blocks(title="Deep-Q Snake — Live Demo") as demo:
    gr.Markdown(
        """
        # 🐍 Deep-Q Snake — Live AI Demo
        A Deep Q-Learning agent (PyTorch) that taught itself to play Snake.
        Press **Watch AI Play** and the trained network drives the snake in real time.

        *Network:* 11 inputs → 256 hidden (ReLU) → 3 outputs · *Actions:* straight / turn right / turn left.
        """
    )
    with gr.Row():
        with gr.Column(scale=3):
            screen = gr.Image(label="Game", height=480, width=640, show_label=False)
        with gr.Column(scale=1):
            speed = gr.Radio(["Slow", "Normal", "Fast"], value="Normal", label="Speed")
            start = gr.Button("▶ Watch AI Play", variant="primary")
            stop = gr.Button("■ Stop")
            status = gr.Textbox(label="Status", interactive=False)

    run = start.click(fn=play, inputs=speed, outputs=[screen, status])
    stop.click(fn=None, inputs=None, outputs=None, cancels=[run])


if __name__ == "__main__":
    import os

    # Bind to 0.0.0.0 and the host-provided $PORT so this works on Render /
    # other PaaS free tiers as well as locally (defaults to 7860).
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
