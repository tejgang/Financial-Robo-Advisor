import torch
import torch.nn as nn

class RiskProfileClassifier(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, output_size=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, output_size)
        )
        
    def forward(self, x):
        return self.net(x) 