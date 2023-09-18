import torch
from data.processed.cross_coupling import datasets
import numpy as np
from utillities.helper import Helper
from models.rbf import RadialBasisFunction
from models.f_cut import CosineCutoff
from models.models import PAINN
from data.processed.cross_coupling import datasets

data = datasets.Cross_coupling_alllig2_test()
data = datasets.Tetris()

# Network, loss function, and optimizer
#net = GNNInvariant(output_dim=data.num_graphs, state_dim = 5)
net = PAINN(num_phys_dims=2)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    output = net(data)
    loss = loss_function(output, data.graph_list)
    loss.backward()
    optimizer.step()

    accuracy = (torch.argmax(output, 1) == data.graph_list).sum() / data.num_graphs
    print(f'Epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}')