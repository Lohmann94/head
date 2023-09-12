import torch
from data.processed.cross_coupling import datasets
import numpy as np
from utillities.helper import Helper

class MetaModel():

    def __init__(self, included_graphs=None, cutoff=None):

        self.dataset = datasets.Cross_coupling_optimized_eom2Lig()
        self.included_graphs = included_graphs
        self.cutoff = cutoff
        self.helper = Helper(self.dataset, self.cutoff, self.included_graphs)

        graph_id = 0
        node_i = 0

        nbrs_i = self.helper.i_nbr(graph_id=graph_id, node_i=node_i)

        
        
        



if __name__ == "__main__":
    meta_model = MetaModel(included_graphs=10, cutoff=10)