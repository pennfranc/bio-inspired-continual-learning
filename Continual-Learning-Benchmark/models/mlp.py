import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, out_dim=10, in_channel=1, img_sz=28, hidden_dim=256, activation_override=None):
        super(MLP, self).__init__()
        self.in_dim = in_channel*img_sz*img_sz
        self.activation_override = activation_override
        self.linear = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            self.activation_function(),
            nn.Linear(hidden_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            self.activation_function()
        )
        self.last = nn.Linear(hidden_dim, out_dim)  # Subject to be replaced dependent on task

    def features(self, x):
        x = self.linear(x.view(-1,self.in_dim))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x

    def activation_function(self):
        if self.activation_override == 'tanh':
            return nn.Tanh()
        else:
            return nn.ReLU(inplace=True)

def MLP20(activation_override=None):
    return MLP(hidden_dim=20, activation_override=activation_override)

def MLP100(activation_override=None):
    return MLP(hidden_dim=100, activation_override=activation_override)

def MLP200(activation_override=None):
    return MLP(hidden_dim=200, activation_override=activation_override)

def MLP400(activation_override=None):
    return MLP(hidden_dim=400, activation_override=activation_override)


def MLP1000():
    return MLP(hidden_dim=1000)


def MLP2000():
    return MLP(hidden_dim=2000)


def MLP5000():
    return MLP(hidden_dim=5000)