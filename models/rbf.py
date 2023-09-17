import torch
import torch.nn as nn
import torch.nn.functional as F


class RadialBasisFunction(nn.Module):
    def __init__(self, n=20, r_cut=20):
        super(RadialBasisFunction, self).__init__()
        self.n = n
        self.r_cut = r_cut

    def forward(self, x):
        # Apply your custom function to the input
        output = self.custom_function(x)
        return output

    def custom_function(self, x):

        values = torch.arange(1,self.n+1)

        activations = torch.zeros(len(values))

        for integer in values:

            activations[integer-1] = torch.sin(((integer*torch.pi)/self.r_cut)*torch.norm(x))/torch.norm(x)

        return activations