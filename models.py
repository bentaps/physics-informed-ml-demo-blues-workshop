import torch
import torch.nn as nn

class VanillaModel(nn.Module):
    """
    """
    def __init__(self, state_dim=1, hidden_dim=128):
        super().__init__()
        
        self.state_dim = state_dim
        self.name = "VanillaModel"
        
        # Learnable spring constant (physics-informed)
        self.spring_coeff = nn.Parameter(torch.tensor([1.0]))
        self.linear_drag_coeff = nn.Parameter(torch.tensor([1.0]))
        
        # Neural network for velocity-dependent forces (damping + quadratic drag)
        self.neural_net = nn.Sequential(
            nn.Linear(2*self.state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )
    
    def forward(self, x, v):
        return self.neural_net(torch.cat((x, v), dim=-1))
    


class PhysicsInformedModel(nn.Module):
    """
    Physics-informed neural network for learning the acceleration field of a 1D pendulum.
    
    Models: dv/dt = f(x, v) = K*x + g(x) + f(v)
    where K is a learnable scalar and g, f are neural networks.
    """
    def __init__(self, state_dim=1, hidden_dim=64):
        super().__init__()
        
        self.state_dim = state_dim
        
        # Learnable spring constant (physics-informed)
        self.spring_coeff = nn.Parameter(torch.tensor([1.0]))
        self.linear_drag_coeff = nn.Parameter(torch.tensor([1.0]))
        
        # Neural network for velocity-dependent forces (damping + quadratic drag)
        self.nonlinear_drag_force = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )
    
    def forward(self, x, v):
        """
        Compute the acceleration field: dv/dt = f(x, v)
        
        Args:
            x: positions [batch_size, state_dim]
            v: velocities [batch_size, state_dim]
        
        Returns:
            acceleration [batch_size, state_dim]
        """
        # Physics-informed structure: dv/dt = -k*x - c*v - f(v)
        spring_force = -self.spring_coeff * x
        drag_force = -self.linear_drag_coeff * v
        nonlinear_drag_force = -self.nonlinear_drag_force(v)

        acceleration = spring_force + drag_force + nonlinear_drag_force
        return acceleration