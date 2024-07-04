# Imports:
# --------
import torch
import torch.nn as nn
from torch.nn import functional as F


# Policy net (pi_theta): Actor
# ----------------------
class PolicyNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()

        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        self.output = nn.Linear(hidden_dim, 4)

    def forward(self, s):
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        outs = self.output(x)

        #! Clamping is done to avoid very high logits
        # return torch.clamp(outs, min=-50, max=+50)
        return outs


# Q-Network: Critic
# ----------
class QNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()

        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        self.output = nn.Linear(hidden_dim, 4)

    def forward(self, s):
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        outs = self.output(x)

        return outs
