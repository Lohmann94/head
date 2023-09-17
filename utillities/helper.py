import numpy as np

class Helper():

    def __init__(self, dataset, cutoff, included_graphs=None):
        self.dataset = dataset
        self.cutoff = cutoff
        self.included_graphs = included_graphs

    def i_nbr(self, graph_id, node_i, direction):
        start_index = list(self.dataset.node_graph_index).index(graph_id)
        end_index = int(start_index + self.dataset.size_graphs[graph_id])

        nodes = self.dataset.node_coordinates[start_index:end_index]

        distances = {}

        for index in range(len(nodes)):
            if index != node_i:
                distances[index] = nodes[index] - nodes[node_i]
        if direction == True:
            nbrs_i = {key:value/np.linalg.norm(value) for key, value in distances.items() if np.linalg.norm(value) <= self.cutoff}
        else:
            nbrs_i = {key:value for key, value in distances.items() if np.linalg.norm(value) <= self.cutoff}


        return nbrs_i

    
