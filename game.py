import pygame
import random
import math
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
label_font = pygame.font.Font('arial.ttf', 13)
value_font = pygame.font.Font('arial.ttf', 20)
value_font.set_bold(True)

# reset
# reward
# play(action) -> direction
# game_iteration
# is_collision

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# color palette — flat-UI dark theme, muted neutrals + green/red accents
BG_DARK = (13, 15, 20)
TILE_DARK = (19, 21, 27)
TILE_LIGHT = (25, 28, 35)
PANEL_BG = (17, 19, 25)
DIVIDER = (40, 43, 51)
TEXT_MUTED = (118, 124, 138)
TEXT_BRIGHT = (232, 234, 238)
ACCENT_GREEN = (46, 204, 113)

# Snake — tapered tail -> head gradient, subtle outline per segment for definition
SNAKE_TAIL = (23, 111, 68)
SNAKE_BODY = (39, 174, 96)
SNAKE_HEAD = (88, 214, 141)
SNAKE_HEAD_SHINE = (170, 240, 200)
SNAKE_OUTLINE = (12, 58, 38)
EYE_DOT = (8, 10, 13)

# Food — warm red, glow layered with per-pixel alpha for a soft heartbeat pulse
FOOD_RED_GLOW = (196, 34, 52)
FOOD_RED_MID = (226, 55, 68)
FOOD_RED_CORE = (255, 84, 92)
FOOD_RED_HOT = (255, 154, 148)

BLOCK_SIZE = 20
# HUD: horizontal stat bar; multiple of BLOCK_SIZE for clean grid below
HUD_HEIGHT = BLOCK_SIZE * 3  # 60px
# Target display FPS for the full train loop (after env step + PyTorch); higher = smoother motion.
TARGET_FPS = 60

def _lerp_rgb(a, b, t):
    t = max(0.0, min(1.0, float(t)))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


class StopTrainingException(Exception):
    """Raised when user requests training to stop via keyboard."""
    pass


class SnakeGameAI:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode(
            (self.w, self.h),
            pygame.DOUBLEBUF,
        )
        pygame.display.set_caption('Snake · RL')
        self.clock = pygame.time.Clock()
        self.generation = 0
        self.reset()
    
    def reset(self):
        self.generation += 1
        # init game state
        self.direction = Direction.RIGHT
        # Snap to grid so head/body/food stay aligned (avoids float drift / weird drawing)
        hx = int(self.w // 2 // BLOCK_SIZE * BLOCK_SIZE)
        hy = int(self.h // 2 // BLOCK_SIZE * BLOCK_SIZE)
        hy = max(hy, HUD_HEIGHT)
        self.head = Point(hx, hy)
        # Use separate Point instances so snake[0] is never the same object as self.head
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
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        # only place food in playfield below HUD
        min_row = HUD_HEIGHT // BLOCK_SIZE
        max_row = (self.h-BLOCK_SIZE )//BLOCK_SIZE
        y = random.randint(min_row, max_row)*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise StopTrainingException()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                raise StopTrainingException()

        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0 
        game_over = False
        stall_timeout = self.frame_iteration > 100 * len(self.snake)
        if self._is_collision() or stall_timeout:
            game_over = True
            reward = -10

            # No red overlay/delay during training (keeps loop smooth; avoids display glitches).
            self._update_ui()
            pygame.display.flip()
            # Frame pacing happens in agent.train() after ML step so timing includes PyTorch work.

            return reward, game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update display (flip was missing here — without it the window barely refreshes during play)
        self._update_ui()
        pygame.display.flip()
        # 6. return game over and score (clock.tick lives in agent loop after train_short_memory)
        return reward, game_over, self.score
    
    def _is_collision(self, pt=None):

        if pt is None:
            pt = self.head
        # hits boundary (top boundary is just below HUD)
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < HUD_HEIGHT:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BG_DARK)

        # Checkerboard playfield (subtle contrast, no grid lines — flatter, cleaner look)
        for x in range(0, self.w, BLOCK_SIZE):
            for y in range(HUD_HEIGHT, self.h, BLOCK_SIZE):
                rect = pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE)
                if (x // BLOCK_SIZE + y // BLOCK_SIZE) % 2 == 0:
                    pygame.draw.rect(self.display, TILE_DARK, rect)
                else:
                    pygame.draw.rect(self.display, TILE_LIGHT, rect)

        nseg = len(self.snake)

        # Snake: rounded, tapering segments with a thin outline for per-segment definition
        for i, pt in enumerate(self.snake):
            px, py = int(pt.x), int(pt.y)
            cell = pygame.Rect(px, py, BLOCK_SIZE, BLOCK_SIZE)
            if i == 0:
                pygame.draw.rect(self.display, SNAKE_HEAD, cell, border_radius=7)
                pygame.draw.rect(self.display, SNAKE_OUTLINE, cell, width=1, border_radius=7)
                # Subtle top/leading shine so head reads clearly
                if self.direction == Direction.RIGHT:
                    pygame.draw.line(
                        self.display, SNAKE_HEAD_SHINE,
                        (cell.left + 3, cell.top + 4), (cell.left + 3, cell.bottom - 4), 2,
                    )
                elif self.direction == Direction.LEFT:
                    pygame.draw.line(
                        self.display, SNAKE_HEAD_SHINE,
                        (cell.right - 4, cell.top + 4), (cell.right - 4, cell.bottom - 4), 2,
                    )
                elif self.direction == Direction.UP:
                    pygame.draw.line(
                        self.display, SNAKE_HEAD_SHINE,
                        (cell.left + 4, cell.bottom - 4), (cell.right - 4, cell.bottom - 4), 2,
                    )
                else:
                    pygame.draw.line(
                        self.display, SNAKE_HEAD_SHINE,
                        (cell.left + 4, cell.top + 3), (cell.right - 4, cell.top + 3), 2,
                    )
                # Eyes, facing current direction
                dot_r = 3
                if self.direction == Direction.RIGHT:
                    centers = [(cell.right - 6, cell.centery - 4), (cell.right - 6, cell.centery + 4)]
                elif self.direction == Direction.LEFT:
                    centers = [(cell.left + 6, cell.centery - 4), (cell.left + 6, cell.centery + 4)]
                elif self.direction == Direction.UP:
                    centers = [(cell.centerx - 4, cell.top + 6), (cell.centerx + 4, cell.top + 6)]
                else:
                    centers = [(cell.centerx - 4, cell.bottom - 6), (cell.centerx + 4, cell.bottom - 6)]
                for c in centers:
                    pygame.draw.circle(self.display, EYE_DOT, c, dot_r)
            else:
                t = (i - 1) / max(nseg - 2, 1)
                body_rgb = _lerp_rgb(SNAKE_TAIL, SNAKE_BODY, t)
                radius = 5 if i < nseg - 1 else 4
                pygame.draw.rect(self.display, body_rgb, cell, border_radius=radius)
                pygame.draw.rect(self.display, SNAKE_OUTLINE, cell, width=1, border_radius=radius)

        self._draw_food()

        # HUD bar: panel distinct from playfield, thin divider with a 2px shadow edge
        pygame.draw.rect(self.display, PANEL_BG, pygame.Rect(0, 0, self.w, HUD_HEIGHT))
        pygame.draw.line(self.display, DIVIDER, (0, HUD_HEIGHT - 2), (self.w, HUD_HEIGHT - 2))
        pygame.draw.line(self.display, (0, 0, 0), (0, HUD_HEIGHT - 1), (self.w, HUD_HEIGHT - 1))

        # Stats: horizontal label/value blocks separated by thin dividers
        stats = [
            ("SCORE", str(self.score), ACCENT_GREEN),
            ("GEN", str(self.generation), TEXT_BRIGHT),
            ("LENGTH", str(len(self.snake)), TEXT_BRIGHT),
        ]
        x = 18
        cy = HUD_HEIGHT // 2
        for i, (label, value, value_color) in enumerate(stats):
            label_surf = label_font.render(label, True, TEXT_MUTED)
            value_surf = value_font.render(value, True, value_color)
            self.display.blit(label_surf, (x, cy - value_surf.get_height() - 1))
            self.display.blit(value_surf, (x, cy + 3))
            x += max(label_surf.get_width(), value_surf.get_width()) + 26
            if i < len(stats) - 1:
                pygame.draw.line(self.display, DIVIDER, (x - 13, 12), (x - 13, HUD_HEIGHT - 12), 1)

    def _draw_food(self):
        # Soft alpha-blended glow (smoother than flat concentric circles), animated as a heartbeat pulse
        fx = int(self.food.x) + BLOCK_SIZE // 2
        fy = int(self.food.y) + BLOCK_SIZE // 2
        beat = 0.5 + 0.5 * math.sin(self.frame_iteration * 0.18)
        pulse = 0.35 + 0.65 * beat
        r_core = max(4, int(5 * pulse) + 3)
        r_glow = r_core + 9

        glow_size = r_glow * 2 + 6
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        center = (glow_size // 2, glow_size // 2)
        pygame.draw.circle(glow_surf, (*FOOD_RED_GLOW, 45), center, r_glow)
        pygame.draw.circle(glow_surf, (*FOOD_RED_MID, 90), center, int(r_glow * 0.65))
        pygame.draw.circle(glow_surf, (*FOOD_RED_CORE, 255), center, r_core)
        pygame.draw.circle(glow_surf, (*FOOD_RED_HOT, 235), (center[0] - 2, center[1] - 2), max(2, r_core // 2))
        self.display.blit(glow_surf, (fx - glow_size // 2, fy - glow_size // 2))

    def _move(self, action):
        # [straight, right, left] — always step from the actual first segment (avoids head/snake desync)
        if len(self.snake) > 0:
            h = self.snake[0]
            self.head = Point(int(h.x), int(h.y))

        a = np.asarray(action, dtype=np.int64).reshape(-1)
        if a.size < 3:
            a = np.array([1, 0, 0], dtype=np.int64)

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(a[:3], np.array([1, 0, 0])):
            new_dir = clock_wise[idx]

        elif np.array_equal(a[:3], np.array([0, 1, 0])):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]

        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = int(self.head.x)
        y = int(self.head.y)
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
