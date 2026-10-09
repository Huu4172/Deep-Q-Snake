"""
Linear_QNet — identical architecture to the original Deep-Q-Snake repo so the
trained weights in model/model.pth load and behave exactly as during training.
Only the parts needed for inference are kept (no QTrainer / training logic).
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def load(self, file_name="model.pth"):
        file_path = os.path.join("./model", file_name)
        if not os.path.exists(file_path):
            return False
        state_dict = torch.load(file_path, map_location="cpu")
        self.load_state_dict(state_dict)
        self.eval()
        return True
