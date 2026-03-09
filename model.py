import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x ):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
        

    def save(self, file_name='model.pth'):
        model_folder_path = './model'

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        model_folder_path = './model'
        file_path = os.path.join(model_folder_path, file_name)

        if not os.path.exists(file_path):
            return False

        state_dict = torch.load(file_path, map_location='cpu')
        self.load_state_dict(state_dict)
        self.eval()
        # ANSI color codes for nicer terminal output
        GREEN = "\033[92m"
        CYAN = "\033[96m"
        YELLOW = "\033[93m"
        MAGENTA = "\033[95m"
        RESET = "\033[0m"

        print(f"{GREEN}Loaded model weights from{RESET} {YELLOW}{file_path}{RESET}")
        for name, param in self.state_dict().items():
            print(f"  {CYAN}{name}{RESET}: shape={MAGENTA}{tuple(param.shape)}{RESET}")
            # Show a small ASCII histogram for weight tensors
            if 'weight' in name and param.numel() > 0:
                data = param.detach().cpu().numpy().ravel()
                hist, bin_edges = np.histogram(data, bins=8)
                max_count = hist.max() if hist.max() > 0 else 1
                bar_width = 30
                print(f"    {GREEN}value distribution:{RESET}")
                for i, count in enumerate(hist):
                    bar = '█' * int(count / max_count * bar_width)
                    print(f"      {YELLOW}{bin_edges[i]: .2f} to {bin_edges[i+1]: .2f}{RESET}: {bar}")
        return True

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.model = model
        self.gamma = gamma
        self.optimiser = optim.Adam(model.parameters(), lr= self.lr)
        self.criterion = nn.MSELoss()
        self.game_count = 0

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype = torch.float)
        action = torch.tensor(action, dtype = torch.long)
        next_state = torch.tensor(np.array(next_state), dtype = torch.float)
        reward = torch.tensor(reward, dtype = torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # Q val of the current state

        pred = self.model(state)
        target = pred.detach().clone()
        
        for i in range(len(done)):
            Q_new = reward[i]

            if not done[i]:
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))

            target[i][torch.argmax(action[i]).item()] = Q_new

        # new_q_val = reward + gamma * max(next_pred_q val)
        #pred.clone()
        #preds[argmax(action)] = Q_new

        self.optimiser.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()

        self.optimiser.step()

        # track completed games and autosave every 100
        for d in done:
            if d:
                self.game_count += 1
                if self.game_count % 100 == 0:
                    print(f"Autosaving model at game {self.game_count}")
                    self.model.save()