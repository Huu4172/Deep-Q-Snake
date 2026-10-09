"""
Headless Snake simulation — a pure-Python/NumPy port of the original pygame
`SnakeGameAI` (game.py) with NO window/display. The grid geometry, movement,
collision rules, food placement and stall-timeout are copied faithfully so the
trained model plays exactly as it did in the original environment.

The 11-value state encoding matches agent.py `Agent.get_state()` exactly.
"""
import random
from enum import Enum
from collections import namedtuple

import numpy as np

# --- constants copied verbatim from the original game.py ---
BLOCK_SIZE = 20
HUD_HEIGHT = BLOCK_SIZE * 3  # 60px reserved at the top for the stats bar


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")


class SnakeGame:
    """Headless equivalent of SnakeGameAI (logic only, no rendering)."""

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.generation = 0
        self.reset()

    def reset(self):
        self.generation += 1
        self.direction = Direction.RIGHT
        hx = int(self.w // 2 // BLOCK_SIZE * BLOCK_SIZE)
        hy = int(self.h // 2 // BLOCK_SIZE * BLOCK_SIZE)
        hy = max(hy, HUD_HEIGHT)
        self.head = Point(hx, hy)
        self.snake = [
            Point(hx, hy),
            Point(hx - BLOCK_SIZE, hy),
            Point(hx - 2 * BLOCK_SIZE, hy),
        ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        min_row = HUD_HEIGHT // BLOCK_SIZE
        max_row = (self.h - BLOCK_SIZE) // BLOCK_SIZE
        y = random.randint(min_row, max_row) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # boundary (top boundary is just below the HUD)
        if (
            pt.x > self.w - BLOCK_SIZE
            or pt.x < 0
            or pt.y > self.h - BLOCK_SIZE
            or pt.y < HUD_HEIGHT
        ):
            return True
        # self collision
        if pt in self.snake[1:]:
            return True
        return False

    def _move(self, action):
        # action: [straight, right-turn, left-turn]
        if len(self.snake) > 0:
            h = self.snake[0]
            self.head = Point(int(h.x), int(h.y))
        a = np.asarray(action, dtype=np.int64).reshape(-1)
        if a.size < 3:
            a = np.array([1, 0, 0], dtype=np.int64)

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(a[:3], np.array([1, 0, 0])):
            new_dir = clock_wise[idx]            # straight
        elif np.array_equal(a[:3], np.array([0, 1, 0])):
            new_dir = clock_wise[(idx + 1) % 4]  # right turn
        else:
            new_dir = clock_wise[(idx - 1) % 4]  # left turn
        self.direction = new_dir

        x, y = int(self.head.x), int(self.head.y)
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)

    def play_step(self, action):
        """Advance one step. Returns (game_over, score)."""
        self.frame_iteration += 1
        self._move(action)
        self.snake.insert(0, self.head)

        game_over = False
        stall_timeout = self.frame_iteration > 100 * len(self.snake)
        if self.is_collision() or stall_timeout:
            game_over = True
            return game_over, self.score

        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        return game_over, self.score

    def get_state(self):
        """11-value state vector, identical ordering to agent.py get_state()."""
        head = self.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # danger straight
            (dir_r and self.is_collision(point_r))
            or (dir_l and self.is_collision(point_l))
            or (dir_u and self.is_collision(point_u))
            or (dir_d and self.is_collision(point_d)),
            # danger right
            (dir_r and self.is_collision(point_d))
            or (dir_l and self.is_collision(point_u))
            or (dir_u and self.is_collision(point_r))
            or (dir_d and self.is_collision(point_l)),
            # danger left
            (dir_r and self.is_collision(point_u))
            or (dir_l and self.is_collision(point_d))
            or (dir_u and self.is_collision(point_l))
            or (dir_d and self.is_collision(point_r)),
            # move direction
            dir_r, dir_l, dir_u, dir_d,
            # food location relative to head
            self.food.x < self.head.x,  # food left
            self.food.x > self.head.x,  # food right
            self.food.y < self.head.y,  # food up
            self.food.y > self.head.y,  # food down
        ]
        return np.array(state, dtype=int)
