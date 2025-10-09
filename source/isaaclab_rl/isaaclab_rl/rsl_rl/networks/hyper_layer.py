import torch
import torch.nn as nn

class HyperLayer(nn.Module):
    def __init__(self, dim, control_dim):
        super(HyperLayer, self).__init__()
        self.dim = dim
        self.control_dim = control_dim

        self.input_weight = nn.Linear(dim, dim, bias=False)
        self.output_weight = nn.Linear(dim, dim, bias=False)
        nn.init.orthogonal_(self.input_weight.weight)
        nn.init.orthogonal_(self.output_weight.weight)
        self.control_proj = nn.Sequential(
            nn.Linear(control_dim, control_dim),
            nn.LayerNorm(control_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(control_dim, dim),
        )

    def forward(self, x, control):
        diag = self.control_proj(control).view(x.shape[0], self.dim)
        return self.output_weight(self.input_weight(x) * diag)
    
    def orthogonal_loss(self):
        eye = torch.eye(self.dim, device=self.input_weight.weight.device)
        in_loss = (self.input_weight.weight @ self.input_weight.weight.T - eye).abs().mean()
        out_loss = (self.output_weight.weight @ self.output_weight.weight.T - eye).abs().mean()
        return in_loss + out_loss