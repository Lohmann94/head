import torch
from models.rbf import RadialBasisFunction
from models.f_cut import CosineCutoff
from models.U_layer import ULayer
from models.V_layer import VLayer
from tqdm import tqdm


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
                 num_message_passing_rounds=1, r_cut=10, n=20,
                 num_phys_dims=3, device='cpu'):
        super().__init__()

        # Define dimensions and other hyperparameters
        self.state_dim = state_dim
        self.edge_dim = 1
        self.output_dim = output_dim
        self.num_message_passing_rounds = num_message_passing_rounds
        self.r_cut = r_cut
        self.n = n
        self.num_phys_dims = num_phys_dims
        self.device = device

        self.phi_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, self.state_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.state_dim, self.state_dim*3),
        ).to(self.device)

        self.filter_net = torch.nn.Sequential(
            RadialBasisFunction(self.n, self.r_cut),
            torch.nn.Linear(self.n, self.state_dim*3),
        ).to(self.device)
        

        # State output network
        self.update_scalar = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim*2, self.state_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.state_dim, self.state_dim*3),
        ).to(self.device)

        # State output network
        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, self.state_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.state_dim, self.output_dim),
        ).to(self.device)

        self.U_layer = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, self.state_dim, bias=False)
        ).to(self.device)

        self.V_layer = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, self.state_dim, bias=False)
        ).to(self.device)

        self.f_cut = CosineCutoff(self.r_cut, self.num_phys_dims)


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

        '''

        self.U_layer = torch.nn.Sequential(
            ULayer(self.state_dim, self.state_dim)
        )

        self.V_layer = torch.nn.Sequential(
            VLayer(self.state_dim, self.state_dim)
        )

        '''

        self.nbr_mask = torch.tensor(
            [index if abs(value) <= self.r_cut else -1 for index, value in enumerate(x.edge_lengths)], device=self.device)

        # Initialize node features to embedding
        self.embedding = torch.nn.Embedding(
            len(x.unique_atoms), self.state_dim, device=self.device)

        self.state = torch.zeros([x.num_nodes, self.state_dim], device=self.device)

        node_graph_index = x.node_graph_index.to(self.device)

        #TODO embedding er fikset, kører igennem hele state som init
        for node in range(node_graph_index.shape[0]):
            self.state[node] = self.embedding(
                torch.tensor(x.atom_to_number[x.node_atoms[node]], device=self.device))

        # Initialize edge vector features to zeros, change when two-dimensional problem
        self.state_vec = torch.zeros(
            [x.num_nodes, self.num_phys_dims, self.state_dim], device=self.device)

        # TODO lav implementation hvor man gør det hele i ét pas i stedet for node for node
        # Loop over message passing rounds

        #TODO spørg: Når man itererer igennem grafer bliver states løbende opdateret
        #TODO dermed influerer det på naboberegningen, men det er jo ikke godt
        #TODO sammenlignet med hvis message og update bliver kørt i ét pas

        for _ in tqdm(range(self.num_message_passing_rounds)):

            for node in tqdm(range(x.node_graph_index.shape[0])):

                # The indices of edges connected with node belonging to graph_i
                #TODO slices er fikset, directions er anderledes afhængig af om det er to eller from
                #TODO Det skal være x.node_to
                edge_slice = torch.where(x.node_from == node)[0]
                edge_slice = edge_slice.to(self.device)

                nbr_i_nodes = torch.tensor(
                    [x.edge_list[value][1] for value in edge_slice if value in self.nbr_mask])
                nbr_i_nodes = nbr_i_nodes.to(self.device)

                nbr_i_edges = torch.tensor(
                    [value for value in edge_slice if value in self.nbr_mask])
                nbr_i_edges = nbr_i_edges.to(self.device)

                # Mask of indeces related to edges for node, with length < r_cut
                nbr_i_states = self.state[nbr_i_nodes]
                nbr_i_states = nbr_i_states.to(self.device)

                edge_vector_diffs = x.edge_vector_diffs.clone().detach()
                edge_vector_diffs = edge_vector_diffs.to(self.device)

                # Getting the edge vector differences (i,j) for the selected edges
                nbrs_cut_diffs = edge_vector_diffs[nbr_i_edges]
                nbrs_cut_diffs = nbrs_cut_diffs.to(self.device)

                """

                The message block:

                """
                # Applying the phi_net to the nbr_i_input tensor
                phi_output = self.phi_net(nbr_i_states)

                # Applying the filter_net to the edge vector differences                
                filter_output = self.filter_net(nbrs_cut_diffs)

                W_output = self.f_cut(filter_output, nbrs_cut_diffs)

                # Normalizing the edge vector differences
                normalized = nbrs_cut_diffs / \
                    torch.norm(nbrs_cut_diffs, dim=1, keepdim=True)

                # Calculating the element-wise product of phi_output and filter_output
                phi_filter_product = phi_output * W_output

                # Splitting the phi_filter_product into three parts
                message_split_1 = phi_filter_product[:, :self.state_dim]
                message_split_2 = phi_filter_product[:,
                                                     self.state_dim:self.state_dim*2]
                message_split_3 = phi_filter_product[:, self.state_dim*2:]

                state_vec_split_1 = self.state_vec[nbr_i_nodes] * \
                    message_split_1[:, None, :]

                normalized_split_3 = normalized[:, :,
                                                None] * message_split_3[:, None, :]

                sum_split_1_and_3 = state_vec_split_1 + normalized_split_3

                # Sum for j
                v_sum_j = torch.sum(sum_split_1_and_3, dim=0)
                s_sum_j = torch.sum(message_split_2, dim=0)

                # Summing over neighbors and updating self.state
                self.state[node] += s_sum_j
                # Summing sum_split_1_and_3 over neighbors and updating self.state_vec
                self.state_vec[node] += v_sum_j

                """

                The Update block:

                """
                # Compute U_product using U_net on state_vec[nbrs_node_from]
                state_vectors = self.state_vec[node].clone().detach()

                U_product = self.U_layer(state_vectors)

                # Compute V_product using V_net on state_vec[nbrs_node_from]
                V_product = self.V_layer(state_vectors)

                # Compute the dot product of U_product and V_product
                UV_product = torch.sum(U_product * V_product, dim=0)

                # Compute the L2 norm of V_product along dimension 1
                V_norm = torch.norm(V_product, dim=0)

                # Concatenate V_norm and self.state[nbrs_node_from] along dimension 1
                combined_tensor = torch.cat(
                    [V_norm, self.state[node]])

                # Compute scalar_pre_split using update_scalar on combined_tensor
                a = self.update_scalar(combined_tensor)

                # Split scalar_pre_split into a_vv, a_sv, and a_ss
                a_vv = a[:self.state_dim]
                a_sv = a[self.state_dim:self.state_dim*2]
                a_ss = a[self.state_dim*2:]

                # Multiply U_product matrix with a_vv matrix
                U_product_a_vv = U_product * a_vv

                # Multiply UV_dot_product matrix with update_split_2 matrix
                UV_dot_product_a_sv = UV_product * a_sv

                sum_a_sv_a_ss = a_ss + UV_dot_product_a_sv

                # Aggregate: Sum messages
                self.state[node] += sum_a_sv_a_ss
                self.state_vec[node] += U_product_a_vv

        # Aggretate: Sum node features
        self.graph_state = torch.zeros((x.num_graphs, self.state_dim)).to(self.device)
        #TODO fiks a node graph index skal være 0,1,2 osv. 
        self.graph_state.index_add_(0, x.node_graph_index.to(self.device), self.state)

        # Output
        out = self.output_net(self.graph_state).flatten()
        return out

class PAINN_2(torch.nn.Module):
    """Translation and rotation invariant graph neural network.

    Keyword Arguments
    -----------------
        output_dim : Dimension of output (default 1)
        state_dim : Dimension of the node states (default 128)
        num_message_passing_rounds : Number of message passing rounds
            (default 3)
    """

    def __init__(self, output_dim=1, state_dim=128,
                 num_message_passing_rounds=1, r_cut=10, n=20,
                 num_phys_dims=3, device='cpu'):
        super().__init__()

        # Define dimensions and other hyperparameters
        self.state_dim = state_dim
        self.edge_dim = 1
        self.output_dim = output_dim
        self.num_message_passing_rounds = num_message_passing_rounds
        self.r_cut = r_cut
        self.n = n
        self.num_phys_dims = num_phys_dims
        self.device = device

        self.phi_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, self.state_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.state_dim, self.state_dim*3),
        ).to(self.device)

        self.filter_net = torch.nn.Sequential(
            RadialBasisFunction(self.n, self.r_cut, self.device),
            torch.nn.Linear(self.n, self.state_dim*3),
        ).to(self.device)
        

        # State output network
        self.update_scalar = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim*2, self.state_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.state_dim, self.state_dim*3),
        ).to(self.device)

        # State output network
        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, self.state_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.state_dim, self.output_dim),
        ).to(self.device)

        self.U_layer = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, self.state_dim, bias=False)
        ).to(self.device)

        self.V_layer = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, self.state_dim, bias=False)
        ).to(self.device)

        self.f_cut = CosineCutoff(self.r_cut, self.num_phys_dims)


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

        '''

        self.U_layer = torch.nn.Sequential(
            ULayer(self.state_dim, self.state_dim)
        )

        self.V_layer = torch.nn.Sequential(
            VLayer(self.state_dim, self.state_dim)
        )

        '''

        self.nbr_mask = torch.tensor(
            [index if abs(value) <= self.r_cut else -1 for index, value in enumerate(x.edge_lengths)], device=self.device)

        # Initialize node features to embedding
        self.embedding = torch.nn.Embedding(
            len(x.unique_atoms), self.state_dim, device=self.device)

        self.state = torch.zeros([x.num_nodes, self.state_dim], device=self.device)

        node_graph_index = x.node_graph_index.to(self.device)

        #TODO embedding er fikset, kører igennem hele state som init
        for node in range(node_graph_index.shape[0]):
            self.state[node] = self.embedding(
                torch.tensor(x.atom_to_number[x.node_atoms[node]], device=self.device))

        # Initialize edge vector features to zeros, change when two-dimensional problem
        self.state_vec = torch.zeros(
            [x.num_nodes, self.num_phys_dims, self.state_dim], device=self.device)

        # TODO lav implementation hvor man gør det hele i ét pas i stedet for node for node
        # Loop over message passing rounds

        #TODO spørg: Når man itererer igennem grafer bliver states løbende opdateret
        #TODO dermed influerer det på naboberegningen, men det er jo ikke godt
        #TODO sammenlignet med hvis message og update bliver kørt i ét pas

        #èt pas loop:
        for _ in tqdm(range(self.num_message_passing_rounds)):
            #OBS ikke reduceret for r_cut

            inp = self.state[x.node_from]

            phi_output = self.phi_net(inp)

            filter_output = self.filter_net(x.edge_vector_diffs)

            W_output = self.f_cut(filter_output, x.edge_vector_diffs.to(self.device))

            normalized = x.edge_vector_diffs / \
                    torch.norm(x.edge_vector_diffs, dim=1, keepdim=True)

            phi_filter_product = phi_output * W_output

            # Splitting the phi_filter_product into three parts
            message_split_1 = phi_filter_product[:, :self.state_dim]
            message_split_2 = phi_filter_product[:,
                                                     self.state_dim:self.state_dim*2]
            message_split_3 = phi_filter_product[:, self.state_dim*2:]

            state_vec_split_1 = self.state_vec[x.node_from] * \
                    message_split_1[:, None, :]

            normalized_split_3 = normalized.to(self.device)[:, :,
                                                None] * message_split_3[:, None, :]

            sum_split_1_and_3 = state_vec_split_1 + normalized_split_3

            self.state.index_add_(0, x.node_to.to(self.device), message_split_2)
            self.state_vec.index_add_(0, x.node_to.to(self.device), sum_split_1_and_3)

            '''
            
            Update block

            '''

            state_vectors = self.state_vec.clone().detach()

            U_product = self.U_layer(state_vectors)

            # Compute V_product using V_net on state_vec[nbrs_node_from]
            V_product = self.V_layer(state_vectors)

            # Compute the dot product of U_product and V_product
            UV_product = torch.sum(U_product * V_product, dim=1)

            # Compute the L2 norm of V_product along dimension 1
            V_norm = torch.norm(V_product, dim=1)

            # Concatenate V_norm and self.state[nbrs_node_from] along dimension 1
            combined_tensor = torch.cat(
                    [V_norm, self.state], dim=1)

            # Compute scalar_pre_split using update_scalar on combined_tensor
            a = self.update_scalar(combined_tensor)

            # Split scalar_pre_split into a_vv, a_sv, and a_ss
            a_vv = a[:,:self.state_dim]
            a_sv = a[:,self.state_dim:self.state_dim*2]
            a_ss = a[:,self.state_dim*2:]

            # Multiply U_product matrix with a_vv matrix
            U_product_a_vv = U_product * \
                    a_vv[:, None, :]

            # Multiply UV_dot_product matrix with update_split_2 matrix
            UV_product_a_sv = UV_product * a_sv

            sum_a_sv_a_ss = a_ss + UV_product_a_sv

            # Aggregate: Sum messages
            self.state += sum_a_sv_a_ss
            self.state_vec += U_product_a_vv

        # Aggretate: Sum node features

        #Index aligned in case data split starts at non zero index:
        index_aligned = x.node_graph_index - x.node_graph_index[0]

        self.graph_state = torch.zeros((x.num_graphs, self.state_dim)).to(self.device)
        self.graph_state.index_add_(0, index_aligned.to(self.device), self.state)

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
