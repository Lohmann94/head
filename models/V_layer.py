import torch
import torch.nn as nn

class VLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(VLayer, self).__init__()
        self.V = nn.Parameter(torch.randn(output_size, input_size))
    
    def forward(self, x):
        return torch.matmul(self.V, x)
