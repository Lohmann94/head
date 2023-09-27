import torch
import torch.nn as nn

class ULayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(ULayer, self).__init__()
        self.U = nn.Parameter(torch.randn(output_size, input_size))
    
    def forward(self, x):
        return torch.matmul(x, self.U)