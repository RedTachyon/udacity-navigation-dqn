import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPModel(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(MLPModel, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.activation = nn.ReLU
        
        mlp_sizes = (state_size, 64, 64)
        
        _mlp_layers = tuple(
            (nn.Linear(in_size, out_size), self.activation())
            for in_size, out_size in zip(mlp_sizes, mlp_sizes[1:])
        )
        self.mlp_layers = nn.Sequential(*sum(_mlp_layers, ()))
        
        self.head = nn.Linear(mlp_sizes[-1], action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.mlp_layers(state)
        return self.head(x)
