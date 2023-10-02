import torch
import torch.nn as nn
import torch.nn.functional as F


class RadialBasisFunction(nn.Module):
    def __init__(self, n=20, r_cut=20):
        super(RadialBasisFunction, self).__init__()
        self.n = n
        self.r_cut = r_cut

        # Check if a GPU is available
        if torch.cuda.is_available():
            print("GPU is available!")
            self.device = torch.device("cuda:0")
        else:
            print("GPU is not available.")
            self.device = torch.device("cpu")


    def forward(self, x):
        # Apply your custom function to the input
        values = torch.arange(1,self.n+1)

        output = torch.zeros(x.shape[0],len(values))
        output = output.to(self.device)
        

        for i in range(x.shape[0]):

            for integer in values:

                output[i][integer - 1] = torch.sin(((integer*torch.pi)/self.r_cut)*torch.norm(x[i]))/torch.norm(x[i])


        return output
