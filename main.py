import torch
from data.processed.cross_coupling import datasets
import numpy as np
from utillities.helper import Helper
from models.rbf import RadialBasisFunction
from models.f_cut import CosineCutoff
from models.models import PAINN
from data.processed.cross_coupling import datasets

data = datasets.Cross_coupling_optimized_eom2Lig()
#data = datasets.Tetris()

# Network, loss function, and optimizer
#net = GNNInvariant(output_dim=data.num_graphs, state_dim = 5)
net = PAINN(num_phys_dims=3)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

torch.autograd.set_detect_anomaly(True)

epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    output = net(data)
    loss = loss_function(output, data.targets)
    loss.backward()
    optimizer.step()

    print(f'Epoch: {epoch}, Loss: {loss}, MSE: {loss.item()}')