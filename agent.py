import torch
import random 
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point, StopTrainingException, TARGET_FPS
from model import Linear_QNet, QTrainer
# Lazy-import helper when training stops so matplotlib does not init at startup (fixes pygame conflicts / stuck red overlay on some systems).


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

SAVE_EVERY_N_GAMES = 10


class Agent():
    def __init__(self):
        self.num_games = 0
        self.epsilon = 0 # controls randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)

        if self.model.load():
            print("Loaded existing model weights from checkpoint.")
        else:
            print("No existing checkpoint found. Starting with a new model.")
        # Checkpoint load uses eval(); switch back for Q-learning updates.
        self.model.train()


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [

            # danger straight
            (dir_r and game._is_collision(point_r)) or
            (dir_l and game._is_collision(point_l)) or
            (dir_u and game._is_collision(point_u)) or
            (dir_d and game._is_collision(point_d)), 

            # right danger
            (dir_r and game._is_collision(point_d)) or
            (dir_l and game._is_collision(point_u)) or
            (dir_u and game._is_collision(point_r)) or
            (dir_d and game._is_collision(point_l)),

            #left danger
            (dir_r and game._is_collision(point_u)) or
            (dir_l and game._is_collision(point_d)) or
            (dir_u and game._is_collision(point_l)) or 
            (dir_d and game._is_collision(point_r)),


            # moving directions

            dir_r,
            dir_l,
            dir_u,
            dir_d,

            # food locations

            game.food.x  < game.head.x, # left
            game.food.x > game.head.x, # right
            game.food.y < game.head.y, # up
            game.food.y > game.head.y #down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
        return loss

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 69 - self.num_games
        final_move = [0,0,0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype = torch.float)
            with torch.no_grad():
                prediction = self.model(state0) 
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    mse_history = []
    episode_rewards = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    episode_reward = 0

    try:
        while True:
            # get the old state 
            state_old = agent.get_state(game)

            # get the move
            final_move = agent.get_action(state_old)

            # perform move and get a new state

            reward, done, score = game.play_step(final_move)
            episode_reward += reward

            # getting the new state
            state_new = agent.get_state(game)

            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember all of this
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # train the long memory and also will plot the result

                game.reset()
                agent.num_games += 1

                loss = agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save()

                if agent.num_games % SAVE_EVERY_N_GAMES == 0:
                    agent.model.save()
                    print(f"Checkpoint saved at game {agent.num_games}")

                print('Game : ', agent.num_games, 'Score : ', score, 'Record : ', record, 'MSE : ', loss)

                # tracking for plots
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.num_games
                plot_mean_scores.append(mean_score)
                episode_rewards.append(episode_reward)
                mse_history.append(loss)

                # reset for next episode
                episode_reward = 0

            # Pace the whole step (env + draw + PyTorch) for smooth motion; tick after ML so
            # heavy train_short_memory is included in the frame budget.
            game.clock.tick(TARGET_FPS)
    except (KeyboardInterrupt, StopTrainingException):
        print("\nTraining interrupted by user. Showing final plots...")
        from helper import show_final_plots

        show_final_plots(plot_scores, plot_mean_scores, episode_rewards, mse_history)





if __name__ == '__main__':
    train()