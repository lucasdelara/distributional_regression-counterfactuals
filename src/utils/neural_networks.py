import torch
import torch.nn as nn
import torch.nn.functional as F

from .basics import split_data_index

# Constrained monotonic neural networks
# Code adapted from: https://github.com/airtai/monotonic-nn
# Original paper: https://proceedings.mlr.press/v202/runje23a/runje23a.pdf 

# Monotonic layers

def apply_monotonicity_indicator(weight, monotonicity_indicator):
  """Change the signs of the coefficients from a weight matrix according to a sign vector.

  Keyword arguments:
  weight -- weight matrix
  monotonicity_indicator -- monotonicity-indicator vector
  """
  weight = weight
  abs_weight = torch.abs(weight)

  # replace original kernel values for positive or negative ones where needed
  w_t = torch.where(monotonicity_indicator==1, abs_weight, weight)
  w_t = torch.where(monotonicity_indicator==-1, -abs_weight, w_t)

  return w_t

def get_activations(activation):
  """
  Take a convex activation function (e.g., ReLU) and return concave and bounded variants.
  """
  def concave_activation(x):
    return -activation(-x)
  def saturated_activation(x):
    cc = activation(torch.ones_like(x)*1.0)
    return 1.0 * torch.where(x <= 0,activation(x + 1.0) - cc,-activation(-(x - 1.0)) + cc)
  return activation, concave_activation, saturated_activation


class LinearMonotonic(nn.Linear):
  """A linear dense monotonic layer (without activation)."""
  def __init__(self,in_features,out_features, monotonicity_indicator, bias = True,device=None,dtype=None):
    super().__init__(in_features, out_features, bias, device, dtype)
    self.monotonicity_indicator = monotonicity_indicator

  def forward(self, input):
    w_t = apply_monotonicity_indicator(self.weight, self.monotonicity_indicator)
    return F.linear(input, w_t, self.bias)

class MonotonicActivation(nn.Module):
  """A monotonic activation layer with convex/concave/saturated expressiveness."""
  def __init__(self, activation_weights = (1.0, 1.0, 1.0), activation_name='ReLU'):
        super().__init__()
        self.activation_weights = activation_weights
        self.activation_name=activation_name
  def forward(self, input):
        units = input.size(dim=1)
        total = self.activation_weights[0]+self.activation_weights[1]+self.activation_weights[2]
        s_convex = round(self.activation_weights[0] * units / total)
        s_concave = round(self.activation_weights[1] * units / total)
        s_saturated = units - s_convex - s_concave
        x_convex, x_concave, x_saturated = torch.split(input, (s_convex, s_concave, s_saturated), dim=-1)
        if self.activation_name=='ReLU':
          convex_activation, concave_activation, saturated_activation = get_activations(F.relu)
        else:
          convex_activation, concave_activation, saturated_activation = get_activations(nn.Identity())
        y_convex = convex_activation(x_convex)
        y_concave = concave_activation(x_concave)
        y_saturated = saturated_activation(x_saturated)

        return torch.cat([y_convex, y_concave, y_saturated], dim=-1)
  
# Predefined neural networks

class MonoNet(nn.Module):
  """An architecture of monotonic neural network."""
  def __init__(self,n_inputs,n_layers,hidden_size,monotonicity_indicator):
        super().__init__()
        self.layers = n_layers
        # Input layer
        self.monotonic_i = LinearMonotonic(in_features=n_inputs,out_features=hidden_size,monotonicity_indicator=monotonicity_indicator)
        self.activation_i = MonotonicActivation(activation_name='ReLU')
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for l in range(n_layers):
          self.hidden_layers.append(LinearMonotonic(in_features=hidden_size,out_features=hidden_size,monotonicity_indicator=torch.ones(hidden_size)))
          self.hidden_layers.append(MonotonicActivation(activation_name='ReLU'))
        # Output layer
        self.monotonic_f = LinearMonotonic(in_features=hidden_size,out_features=1,monotonicity_indicator=torch.ones(hidden_size))

  def forward(self,x):
        out = self.monotonic_i(x)
        out = self.activation_i(out)
        for layer in self.hidden_layers:
          out = layer(out)
        out = self.monotonic_f(out)
        return out

class Net(nn.Module):
  """A standard neural network."""
  def __init__(self,n_inputs,n_layers,hidden_size):
        super().__init__()
        self.layers = n_layers
        # Input layer
        self.linear_i = nn.Linear(in_features=n_inputs,out_features=hidden_size)
        self.activation_i = nn.ReLU()
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for l in range(n_layers):
          self.hidden_layers.append(nn.Linear(in_features=hidden_size,out_features=hidden_size))
          self.hidden_layers.append(nn.ReLU())
        # Output layer
        self.linear_f = nn.Linear(in_features=hidden_size,out_features=1)

  def forward(self,x):
        out = self.linear_i(x)
        out = self.activation_i(out)
        for layer in self.hidden_layers:
          out = layer(out)
        out = self.linear_f(out)
        return out