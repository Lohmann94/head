import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineCutoff(nn.Module):
    def __init__(self, r_cut=2.0, num_phys_dims=3):
        super(CosineCutoff, self).__init__()
        self.r_cut = r_cut
        self.num_phys_dims = num_phys_dims

    def forward(self, x, r_ij):
        #TODO spørg mikkel om måden at gange r_ij på er rigtig
        output = 0.5 * (torch.cos((torch.pi * x) / self.r_cut) + 1) * sum(r_ij).repeat(int(384/self.num_phys_dims))
        return output

        