import torch
import torch.nn as nn

class ULayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(ULayer, self).__init__()
        self.U = nn.Parameter(torch.randn(output_size, input_size))
    
    def forward(self, x):
        return torch.matmul(self.U, x)

# Example usage
input_size = 128
output_size = 64

model = MyModel(input_size, output_size)
input_tensor = torch.randn(input_size)

output_tensor = model(input_tensor)