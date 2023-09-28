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
        #regn længden af vektorerne, og skal ind på x's plads
        output = 0.5 * (torch.cos((torch.pi * torch.norm(r_ij, dim=1, keepdim=True)) / self.r_cut) + 1) * x
        return output

        