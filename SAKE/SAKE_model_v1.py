# Imports
import torch
import numpy as np
import math
from torch_cluster import radius_graph
from torch_scatter import scatter, scatter_add

#######################################################################################################################################
##############################               Define expnorm RBF function from TorchMD-Net                ##############################
#######################################################################################################################################
class expnorm_smearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=32):
        super().__init__()
        offset = torch.linspace(torch.exp(-torch.tensor(stop)), 1, num_gaussians)  # Determines the center of each function
        self.beta = torch.pow(2*torch.pow(torch.tensor(num_gaussians), -1.)*(1 - torch.exp(-torch.tensor(stop))), -2.)  # Determines the width of each function
        self.register_buffer('offset', offset)

    def forward(self, dist):
        return torch.exp(-self.beta * torch.pow(torch.exp(-dist.view(-1, 1)) - self.offset.view(1, -1), 2))
    
#######################################################################################################################################
##############################               Define SAKE Layer from Wang & Chodera, 2023.                ##############################
#######################################################################################################################################
class SAKELayer(torch.nn.Module):
    """
    SAKE Layer, implemented based on code from
    E(n) Equivariant Convolutional Layer
    """

    def __init__(self, 
                 input_nf, 
                 output_nf, 
                 hidden_nf, 
                 edges_in_d=0,
                 act_fn=torch.nn.CELU(alpha=2.0), 
                 n_heads=4, 
                 cutoff=1,
                 kernel_size=50,
                 normalize=True
                ):
        
        super(SAKELayer, self).__init__()
        input_edge = input_nf * 2
        self.normalize = normalize
        self.epsilon = 1e-8   # Add when dividing by parameters to prevent divide by 0
        edge_coords_nf = 1
        self.cutoff = cutoff
        self.n_heads = n_heads
        
        # Modefied for SAKE (hidden_nf*2)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(input_edge + edge_coords_nf + hidden_nf + edges_in_d, hidden_nf),
            act_fn,
            torch.nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )
        
        # Modified for SAKE
        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_nf + input_nf + hidden_nf, hidden_nf),
            act_fn,
            torch.nn.Linear(hidden_nf, output_nf),
            act_fn
        )

        # Add attention heads based on n_heads
        self.spatial_att_mlp = torch.nn.Linear(hidden_nf, self.n_heads)
        
        # Define semantic attention model
        '''
        Note -  this is not the same as the model used in the original SAKE paper.
        See SI of our paper for details.
        '''
        self.semantic_att_mlp = torch.nn.Sequential(torch.nn.Linear(hidden_nf, self.n_heads),
                                                    torch.nn.CELU(alpha=2.0),
                                                    torch.nn.Linear(self.n_heads, 1)
                                                   )

        # Radial basis function
        self.rbf = torch.nn.Sequential(expnorm_smearing(stop=self.cutoff, num_gaussians=kernel_size),
                                       torch.nn.Linear(kernel_size, hidden_nf)
                                      )
        
        # Filter generating network
        self.filter_nn = torch.nn.Sequential(torch.nn.Linear(hidden_nf*2, hidden_nf*2), 
                                             act_fn, 
                                             torch.nn.Linear(hidden_nf*2, hidden_nf))
        
        # Mu network from spatial attention
        self.mu = torch.nn.Sequential(torch.nn.Linear(self.n_heads, hidden_nf), 
                                      act_fn, 
                                      torch.nn.Linear(hidden_nf, hidden_nf),
                                      act_fn)
        
    #######################################################################################
    ###########          Initialization finished, define sub-models           #############
    #######################################################################################
        
    # Edge featurization model, modified for SAKE
    def edge_model(self, source, target, radial):
        # Get RBF
        rbf = self.rbf(radial)
        # Pass concatenated node embeddings through filter nn
        W = self.filter_nn(torch.cat([source, target], dim=1))
        # Concatenate all features for edge embedding
        out = torch.cat([source, target, radial, rbf*W], dim=1)
        out = self.edge_mlp(out)
        return out
    
    
    # SAKE spatial attention model
    def spatial_attention(self, x, edge_idx, coord_diff, edge_attr):
        row, col = edge_idx
        # Normalize coord_diff
        coord_diff = coord_diff / coord_diff.norm(dim=1).view(-1, 1) + self.epsilon
        # Reshape coord_diff for multiplication w/ attn weights   (n_edges, n_heads, n_coord_dims)
        coord_diff = torch.repeat_interleave(coord_diff.unsqueeze(dim=1), self.n_heads, dim=1)
        
        # Spatial attention
        # Reshape for multiplication w/ coord_diff   (n_edges, n_heads, 1)
        attn = self.spatial_att_mlp(edge_attr).unsqueeze(dim=2)
        attn = attn * coord_diff
        
        # Aggregate across edges and attn heads
        all_aggs = scatter_add(attn, row.unsqueeze(1), dim=0)
        
        # Input to mu MLP
        out = self.mu(torch.norm(all_aggs, dim=2))
        return out, all_aggs
    
    # Define cosine cutoff function
    def cosine_cutoff(self, dist):
        C = 0.5 * (torch.cos(dist * torch.tensor(math.pi) / self.cutoff) + 1.0)
        return C
    
    # Define distance and semantic attention
    def dist_x_semantic_attn(self, radial, edge_attr):
        # Distance-based attention
        '''
        NOTE - if going beyond 1 nm cutoff, or if using units of angstroms,
        change line below to the following:
        
        euclidean_att = self.cosine_cutoff(radial.sqrt())
        
        '''
        euclidean_att = self.cosine_cutoff(radial)
        
	    # Semantic attention
        semantic_att = self.semantic_att_mlp(edge_attr) # Output same shape as edge embedding, perform element-wise mult w/ edges
        
        return semantic_att * euclidean_att

    
    # Node featurization
    def node_model(self, x, edge_index, edge_attr, spatial_att):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        agg = torch.cat([x, agg, spatial_att], dim=1)
        out = self.node_mlp(agg)
        out = x + out
        return out
    

    # Compute pairwise distances and distance vectors
    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff
    

    # Forward function for SAKE layer
    def forward(self, h, edge_index, coord, batch):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial)
        edge_feat = edge_feat * self.dist_x_semantic_attn(radial, edge_feat)
        
        spat_attn, all_aggs = self.spatial_attention(h, edge_index, coord_diff, edge_feat)
        h = self.node_model(h, edge_index, edge_feat, spat_attn)

        return h

#######################################################################################################################################
##############################                 Define scatter-add function used in EGNN                  ##############################
#######################################################################################################################################
def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

##############################################################################################################################
##################                 Define shifted softplus activation function from SchNet                  ##################
##############################################################################################################################
class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return torch.nn.functional.softplus(x) - self.shift
    
    
##############################################################################################################################
###################           Define SAKE model with an additional two-layer energy-predicting NN          ###################
##############################################################################################################################
class SAKE(torch.nn.Module):
    def __init__(self, 
                 in_node_nf, 
                 hidden_nf, 
                 out_node_nf, 
                 in_edge_nf=0, 
                 device='cpu', 
                 act_fn=torch.nn.CELU(alpha=2.0), 
                 energy_act_fn=torch.nn.CELU(alpha=2.0), 
                 n_layers=4,
                 n_heads=4,
                 cutoff=1,
                 kernel_size=18,
                 max_num_neighbors=10000,
                 embed_type='c36',
                 normalize=False, 
                ):
        '''

        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features (not used here)
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param energy_act_fn: Non-linearity for the energy-predicting NN
        :param n_layers: Number of layer for the SAKE
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    This option is from EGNN code, not used in our work.
        '''
        
        super(SAKE, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.cutoff = cutoff
        self.kernel_size = kernel_size
        self.max_num_neighbors = max_num_neighbors
        
        # Set num_types based on embedding type
        '''
        NOTE - this can (and should) be modified to account
        for any desired embedding type.
        '''
        if embed_type == 'c36':
            num_types = 41
        elif embed_type == 'ff14SB':
            num_types = 47
        elif embed_type == 'gaff':
            num_types = 97
        elif embed_type == 'elements':
            num_types = 10
        elif embed_type == 'names':
            num_types = 83
        else:
            raise ValueError ('Invalid embedding type, must be "c36", "ff14SB", "gaff", "elements", or "names".')

        self.embedding_in = torch.nn.Embedding(num_types, self.hidden_nf)
        self.embedding_out = torch.nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("SAKE_%d" % i, SAKELayer(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, n_heads=self.n_heads, cutoff=self.cutoff, kernel_size=self.kernel_size, normalize=normalize))
        
        # Define feed forward network that predicts energy contribution for each output
        self.energy_network = torch.nn.Sequential(torch.nn.Linear(out_node_nf, out_node_nf//2),
                                                  energy_act_fn,
						  torch.nn.Linear(out_node_nf//2, out_node_nf//4),
						  energy_act_fn,
                                                  torch.nn.Linear(out_node_nf//4, 1)
                                                 )
        
        self.to(self.device)

    def forward(self, h, x, batch):
        # Move necessary things to the device of choice
        x = x.to(self.device)
        batch = batch.to(self.device)
        h = h.to(self.device)
        
        # Generate adjacency lists
        edges = radius_graph(x, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors)
        
        # Run SAKE
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h = self._modules["SAKE_%d" % i](h, edges, x, batch)
        
        h = self.embedding_out(h)
        
        # Run the energy predition network
        h = self.energy_network(h)
        
        # Sum pooling
        out = scatter(h, batch, dim=0, reduce='add')
        
        #return h, x
        return out.squeeze()
    
##############################################################################################################################
### Create SAKE layers
##############################################################################################################################
def create_SAKE_layers(in_node_nf, 
                       hidden_nf, 
                       out_node_nf, 
                       in_edge_nf=0, 
                       act_fn=torch.nn.CELU(alpha=2.0), 
                       energy_act_fn=torch.nn.CELU(alpha=2.0), 
                       n_layers=4,
                       n_heads=4,
                       cutoff=1,
                       kernel_size=18,
                       embed_type = 'c36',
                       normalize=False, 
                      ):
    
    # Set num_types based on embedding type
    '''
    NOTE - this can (and should) be modified to account
    for any desired embedding type.
    '''
    if embed_type == 'c36':
        num_types = 41
    elif embed_type == 'ff14SB':
        num_types = 47
    elif embed_type == 'gaff':
        num_types = 97
    elif embed_type == 'elements':
        num_types = 10
    elif embed_type == 'names':
        num_types = 83
    else:
        raise ValueError ('Invalid embedding type, must be "c36", "ff14SB", "gaff", "elements", or "names".')

    # Create embedding_in layer
    embedding_in = torch.nn.Embedding(num_types, hidden_nf)

    # Create embedding_out layer
    embedding_out = torch.nn.Linear(hidden_nf, out_node_nf)
    
    # Create SAKE message passing layers
    sake_conv = torch.nn.ModuleList()
    for _ in range(n_layers):
        conv = SAKELayer(hidden_nf, hidden_nf, hidden_nf, edges_in_d=in_edge_nf,
                         act_fn=act_fn, n_heads=n_heads, cutoff=cutoff, kernel_size=kernel_size, normalize=normalize
                        )
        
        sake_conv.append(conv)
        
    # Create energy-predicting NN
    energy_network = torch.nn.Sequential(torch.nn.Linear(out_node_nf, out_node_nf//2),
                                         energy_act_fn,
					 torch.nn.Linear(out_node_nf//2, out_node_nf//4),
					 energy_act_fn,
                                         torch.nn.Linear(out_node_nf//4, 1)
                                        )
    
    # Return all layers
    return embedding_in, embedding_out, sake_conv, energy_network

##############################################################################################################################
###################           Define modular SAKE model with an additional two-layer energy-predicting NN          ###################
##############################################################################################################################
class SAKE_modular(torch.nn.Module):
    def __init__(self, 
                 embedding_in, 
                 embedding_out, 
                 sake_conv, 
                 energy_network, 
                 device='cpu', 
                 cutoff=1,
                 max_num_neighbors=10000,
                 normalize=False, 
                ):
        '''

        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features (not used here)
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param energy_act_fn: Non-linearity for the energy-predicting NN
        :param n_layers: Number of layer for the SAKE
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    This option is from EGNN code, not used in our work.
        '''
        
        super(SAKE_modular, self).__init__()
        self.device = device
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        
        # Define layers
        self.embedding_in = embedding_in
        self.embedding_out = embedding_out
        self.sake_conv = sake_conv
        self.energy_network = energy_network
        
        # Move model to GPU
        self.to(self.device)

    def forward(self, h, x, batch):
        # Move necessary things to the device of choice
        x = x.to(self.device)
        batch = batch.to(self.device)
        h = h.to(self.device)
        
        # Generate adjacency lists
        edges = radius_graph(x, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors)
        
        # Run SAKE
        h = self.embedding_in(h)
        
        for interaction in self.sake_conv:
            h = interaction(h, edges, x, batch)
        
        h = self.embedding_out(h)
        
        # Run the energy predition network
        h = self.energy_network(h)
        
        # Sum pooling
        out = scatter(h, batch, dim=0, reduce='add')
        
        #return h, x
        return out.squeeze()
