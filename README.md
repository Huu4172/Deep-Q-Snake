# Deep-Q Snake

A Deep Q-Learning agent that teaches itself to play Snake, built with PyTorch and pygame.

## How it works
- **Network:** `Linear_QNet`, 11 inputs → 256 hidden (ReLU) → 3 outputs, one Q-value per move.
- **State (11 inputs):** danger straight / right / left · current heading (4) · food direction (4).
- **Actions:** go straight, turn right, turn left.
- **Rewards:** +10 for eating food, −10 for dying (or stalling too long), 0 otherwise.
- **Training:** epsilon-greedy exploration that decays over the first ~70 games, a replay memory of 100k steps, and a short-memory update every step plus a 1,000-sample batch after each game. Discount γ = 0.9, Adam with lr = 0.001.

## Files
| File | What it does |
|---|---|
| `game.py` | pygame Snake environment (`SnakeGameAI`) |
| `model.py` | `Linear_QNet` and `QTrainer` (the Q-learning update) |
| `agent.py` | state encoding, action selection, replay memory, training loop |
| `helper.py` | training graphs: score, reward distribution, loss |
| `model/model.pth` | trained weights (loaded automatically by `agent.py`) |

## Run it
```bash
pip install -r requirements.txt
python agent.py    # press Q to stop and see the training graphs
```

`game.py` uses `arial.ttf` for the score bar. The font isn't included here because its license doesn't allow redistribution; copy any `.ttf` font into the project folder as `arial.ttf`.
