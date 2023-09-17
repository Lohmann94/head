import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineCutoff(nn.Module):
    def __init__(self, f_cut=2.0):
        super(CosineCutoff, self).__init__()
        self.f_cut = f_cut

    def forward(self, x):
        # Apply your custom function to the input
        output = self.custom_function(x)
        return output

    def custom_function(self, x):

        return 0.5 * (torch.cos((torch.pi * x) / self.f_cut) + 1)