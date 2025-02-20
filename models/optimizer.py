import torch.nn as nn

class PortfolioOptimizer(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, output_size=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1) 
        )
        
    def forward(self, x):
        return self.net(x) 