import torch
from models.rbf import RadialBasisFunction
from models.f_cut import CosineCutoff
from models.U_layer import ULayer
from models.V_layer import VLayer

class PAINN(torch.nn.Module):
    """Translation and rotation invariant graph neural network.

    Keyword Arguments
    -----------------
        output_dim : Dimension of output (default 1)
        state_dim : Dimension of the node states (default 128)
        num_message_passing_rounds : Number of message passing rounds
            (default 3)
    """

    def __init__(self, output_dim=1, state_dim=128,
                 num_message_passing_rounds=3, r_cut=4, n=20,
                 f_cut=2, num_phys_dims=3):
        super().__init__()

        # Define dimensions and other hyperparameters
        self.state_dim = state_dim
        self.edge_dim = 1
        self.output_dim = output_dim
        self.num_message_passing_rounds = num_message_passing_rounds
        self.r_cut = r_cut
        self.f_cut = f_cut
        self.n = n
        self.num_phys_dims = num_phys_dims

        self.phi_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, self.state_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.state_dim, self.state_dim*3),
        )

        self.filter_net = torch.nn.Sequential(
            RadialBasisFunction(self.n, self.r_cut),
            torch.nn.Linear(self.n, self.state_dim*3),
            CosineCutoff(self.f_cut)
        )

        #TODO spørg mikkel om dimensionerne her. Vektor: nbrs x num_phys_dimension x state_dim (128)
        self.U_layer = torch.nn.Sequential(
            ULayer((x.num_nodes, self.num_phys_dims, self.state_dim ), (x.num_nodes,self.num_phys_dims, self.state_dim)),
        )

        #TODO spørg mikkel om dimensionerne her. Vektor: nbrs x num_phys_dimension x state_dim (128)
        self.V_layer = torch.nn.Sequential(
            VLayer((x.num_nodes, self.num_phys_dims, self.state_dim ), (x.num_nodes,self.num_phys_dims, self.state_dim)),
        )

        # State output network
        self.update_scalar = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim*2, self.state_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.state_dim, self.output_dim*3),
        )

        # State output network
        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, self.state_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.state_dim, self.output_dim),
        )

    def forward(self, x):
        """Evaluate neural network on a batch of graphs.

        Parameters
        ----------
        x: GraphDataset
            A data set object that contains the following member variables:

            node_coordinates : torch.tensor (N x 2)
                2d coordinates for each of the N nodes in all graphs
            node_graph_index : torch.tensor (N)
                Index of which graph each node belongs to
            edge_list : torch.tensor (E x 2)
                Edges (to-node, from-node) in all graphs
            node_to, node_from : Shorthand for column 0 and 1 in edge_list
            edge_lengths : torch.tensor (E)
                Edge features
            edge_vectors : torch.tensor (E x 2)
                Edge vector features

        Returns
        -------
        out : N x output_dim
            Neural network output


        # Initialize node features to zeros
        self.state = torch.zeros([x.num_nodes, self.state_dim])

        # Initialize edge vector features to zeros, change when two-dimensional problem
        self.state_vec = torch.zeros([x.num_nodes, 3, self.state_dim])

        """

        self.nbr_mask = torch.tensor(
            [True if abs(value) <= self.r_cut else False for value in x.edge_lengths])


        # Initialize node features to embedding
        # TODO: Fikse at det bliver initialiseret til en embedding 
        # TODO: I PAINN embedder de på baggrund ac kerne-energier (Nuclear charges Z_i) Hvad embedder vi på baggrund af her?

        """
        Eksempel på embedding:

        import torch

        # Define the vocabulary size and embedding dimension
        vocab_size = 100
        embedding_dim = 50

        # Create an instance of the Embedding layer
        embedding = torch.nn.Embedding(vocab_size, embedding_dim)

        # Generate some random input indices
        input_indices = torch.tensor([[1, 2, 3], [4, 5, 6]])

        # Pass the input indices through the Embedding layer
        embedded_output = embedding(input_indices)

        # Print the shape of the embedded output
        print(embedded_output.shape)
        """
        self.state = torch.zeros([x.num_nodes, self.state_dim])

        # Initialize edge vector features to zeros, change when two-dimensional problem
        # TODO: Er det rigtigt forstået at det skal initialiseres til 3 for et 3d vektor rum
        self.state_vec = torch.zeros([x.num_nodes, self.num_phys_dims, self.state_dim])

        # Loop over message passing rounds
        for _ in range(self.num_message_passing_rounds):

            for node in range(x.node_graph_index.shape[0]):

                # Storing the graph index of the current node
                graph_i = x.node_graph_index[node]
                
                # The indices of edges connected with node belonging to graph_i
                edge_slice = torch.where(x.node_from == node)[0]
                
                # Mask of indeces related to edges for node, with length < r_cut
                sub_nbr_mask = torch.index_select(self.nbr_mask, 0, edge_slice)
                
                # Getting the edge vector differences (i,j) for the selected edges
                sub_edge_vector_diffs = x.edge_vector_diffs[edge_slice]
                
                # Getting the 'from' nodes for the selected edges
                sub_node_from = x.node_from[edge_slice]
                
                # Getting the 'to' nodes for the selected edges
                sub_node_to = x.node_to[edge_slice]
                
                # Filtering the edge vector differences based on the sub_nbr_mask
                nbrs_cut_diffs = sub_edge_vector_diffs[sub_nbr_mask]
                
                # Filtering the 'from' nodes based on the sub_nbr_mask
                nbrs_node_from = sub_node_from[sub_nbr_mask]
                
                # Filtering the 'to' nodes based on the sub_nbr_mask
                nbrs_node_to = sub_node_to[sub_nbr_mask]
                
                # Getting the input tensor for the phi_net
                input_tensor = self.state[nbrs_node_from]
                
                # Applying the phi_net to the input tensor
                phi_output = self.phi_net(input_tensor)
                
                # Applying the filter_net to the edge vector differences
                filter_output = self.filter_net(nbrs_cut_diffs)
                
                # Normalizing the edge vector differences
                normalized = nbrs_cut_diffs / torch.tensor(torch.norm(nbrs_cut_diffs))
                
                # Calculating the element-wise product of phi_output and filter_output
                phi_filter_product = phi_output * filter_output
                
                # Splitting the phi_filter_product into three parts
                message_split_1 = phi_filter_product[:, :self.state_dim]
                message_split_2 = phi_filter_product[:, self.state_dim:self.state_dim*2]
                message_split_3 = phi_filter_product[:, self.state_dim*2:]
                
                #TODO Spørg mikkel ift dimensionerne her 3 nbrs, 2 dimensioner, 128 state_dim og 3 nbrs x 128 state_dim
                state_vec_split_1 = self.state_vec[nbrs_node_from] * message_split_1[:, None, :]

                #TODO Spørg mikkel ift dimensionerne her 3 nbrs, 2 koordinater og 3 nbrs x 128 state_dim
                normalized_split_3 = normalized * message_split_3
                
                sum_split_1_and_3 = state_vec_split_1 + normalized_split_3
                
                # Summing over neighbors and updating self.state
                self.state.index_add_(0, nbrs_node_to, message_split_2)
                
                # Summing sum_split_1_and_3 over neighbors and updating self.state_vec
                self.state_vec.index_add_(0, nbrs_node_to, sum_split_1_and_3)

                """
                The Update block:
                """

                # Compute U_product using U_net on state_vec[nbrs_node_from]
                U_product = self.U_net(self.state_vec[nbrs_node_from])
                
                # Compute V_product using V_net on state_vec[nbrs_node_from]
                V_product = self.V_net(self.state_vec[nbrs_node_from])
                
                # Compute the dot product of U_product and V_product
                UV_dot_product = torch.dot(U_product, V_product)
                
                # Compute the L2 norm of V_product along dimension 1
                V_norm = torch.norm(V_product, dim=1)
                
                # Concatenate V_norm and self.state[nbrs_node_from] along dimension 1
                combined_tensor = torch.cat([V_norm, self.state[nbrs_node_from]], dim=1)
                
                # Compute scalar_pre_split using update_scalar on combined_tensor
                scalar_pre_split = self.update_scalar(combined_tensor)
                
                # Split scalar_pre_split into a_vv, a_sv, and a_ss
                a_vv = scalar_pre_split[:, :self.state_dim]
                a_sv = scalar_pre_split[:, self.state_dim:self.state_dim*2]
                a_ss = scalar_pre_split[:, self.state_dim*2:]

                # Multiply U_product matrix with update_split_1 matrix
                U_product_split_1 = U_product[:, :self.state_dim] * update_split_1[:, None, :]
                
                # Multiply UV_dot_product matrix with update_split_2 matrix
                UV_dot_product_split_2 = UV_dot_product[:, :self.state_dim] * update_split_2[:, None, :]

                # Aggregate: Sum messages
                self.state.index_add_(0, x.node_to, message)
                self.state_vec.index_add_(0, x.node_to, message_vec)

        # Aggretate: Sum node features
        self.graph_state = torch.zeros((x.num_graphs, self.state_dim))
        self.graph_state.index_add_(0, x.node_graph_index, self.state)

        # Output
        out = self.output_net(self.graph_state)
        return out


def cross(v1, v2):
    """Compute the 2-d cross product.

    Parameters
    ----------
    v1, v2 : Array
        Arrays (shape Nx2) containing N 2-d vectors

    Returns
    -------
    Array
        Array (shape N) containing the cross products

    """
    return v1[:, 0]*v2[:, 1] - v1[:, 1]*v2[:, 0]


def dot(v1, v2):
    """Compute the 2-d dot product.

    Parameters
    ----------
    v1, v2 : Array
        Arrays (Nx2) containing N 2-d vectors

    Returns
    -------
    Array
        Array (shape N) containing the dot products

    """
    return v1[:, 0]*v2[:, 0] + v1[:, 1]*v2[:, 1]
