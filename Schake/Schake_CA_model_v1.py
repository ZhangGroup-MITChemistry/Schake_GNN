# Imports
import torch
import numpy as np
import math
from torch_scatter import scatter, scatter_add

# Import ML libraries
from torch.utils.data import DataLoader
from torch_geometric.data.makedirs import makedirs
from torch_geometric.nn import MessagePassing, radius_graph

# Other misc. libraries
#import warnings

############################################################################################################################
############################################################################################################################
# Define the Schake model
############################################################################################################################
############################################################################################################################

class Schake_modular(torch.nn.Module):
    def __init__(self,
                 embedding_in,
                 embedding_out,
                 sake_rbf_func,
                 schnet_rbf_func,
                 sake_layers,
                 schnet_layers,
                 sake_low_cut,
                 sake_high_cut,
                 schnet_low_cut,
                 schnet_high_cut,
                 energy_network,
                 normalize,
                 max_num_neigh = 10000,
                 h_schnet = 1,
                 device = 'cpu',
                ):
        
        super(Schake_modular, self).__init__()
        self.device = device
        self.normalize = normalize
        self.max_num_neigh = max_num_neigh
        
        # Define layers
        self.embedding_in = embedding_in
        self.embedding_out = embedding_out
        self.sake_rbf_func = sake_rbf_func
        self.schnet_rbf_func = schnet_rbf_func
        self.sake_layers = sake_layers
        self.schnet_layers = schnet_layers
        self.energy_network = energy_network
        
        # Define cutoffs
        self.sake_low_cut = sake_low_cut
        self.sake_high_cut = sake_high_cut
        self.schnet_low_cut = schnet_low_cut
        self.schnet_high_cut = schnet_high_cut
        
        # Define atom type to mask
        self.h_schnet = h_schnet
        
        # Move model to device
        self.to(self.device)
        
    # Compute pairwise distances and distance vectors
    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff
        
    def forward(self, h, x, batch):
        # Move necessary things to the device of choice
        x = x.to(self.device)
        batch = batch.to(self.device)
        h = h.to(self.device)
        
        # Generate adjacency lists
        edges = radius_graph(x, 
                             r=self.schnet_high_cut, # This can be different depending on which model
                             batch=batch, 
                             max_num_neighbors=self.max_num_neigh  # Must include all possible pairs
                            )
        
        # Compute pairwise distances and edge vectors
        radial, coord_diff = self.coord2radial(edges, x)
        
        # Compute distances (sqrt of radial)
        dist = torch.sqrt(radial)
        
        # Filter edges, coord_diff, radial, dist, rbf based on individual cutoffs
        sake_mask = torch.where((dist < self.sake_high_cut) & (dist >= self.sake_low_cut))[0]
        schnet_mask = torch.where((dist >= self.schnet_low_cut) & (dist <= self.schnet_high_cut))[0]
        
        # Reshape the edges, extract only necessary edges for each model
        sake_edges = edges.T[sake_mask].T
        schnet_edges = edges.T[schnet_mask].T
        
        # Extract radial, coord_diff for SAKE pairs only
        sake_radial, sake_coord_diff = radial[sake_mask], coord_diff[sake_mask]
        
        # Extract distance for SchNet pairs only
        schnet_dist = dist[schnet_mask]
        
        # If h_schnet defined, filter only atom type of interest
        if self.h_schnet != None:
            
            # For SchNet pairs, create adjacency list in terms of species
            h_schnet_edges = h[schnet_edges]
            
            # Filter SchNet edges to only include atom type of interest
            h_mask = torch.where(h_schnet_edges[0] == self.h_schnet)[0]
            schnet_edges = schnet_edges.T[h_mask].T

            # Extract distance for SchNet pairs only
            schnet_dist = schnet_dist[h_mask]
        
        # Create radial basis functions
        sake_rbf = self.sake_rbf_func(sake_radial)
        schnet_rbf = self.schnet_rbf_func(schnet_dist)
        
        # Generate initial embedding
        h = self.embedding_in(h)
        
        # Run layers (SAKE for short-range, SchNet for long)
        for sake_int, schnet_int in zip(self.sake_layers, self.schnet_layers):
            h = h + sake_int(h, sake_edges, sake_radial, sake_coord_diff, sake_rbf)
            h = h + schnet_int(h, schnet_edges, schnet_dist, schnet_rbf)
            
        # Run embedding out layer
        h = self.embedding_out(h)
            
        # Run energy model
        h = self.energy_network(h)
        
        # Sum pooling
        out = scatter(h, batch, dim=0, reduce='add')
        return out.squeeze()
    
    
############################################################################################################################
############################################################################################################################
# Define expnorm smearing function
############################################################################################################################
############################################################################################################################

class expnorm_smearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=32):
        super().__init__()
        offset = torch.linspace(torch.exp(-torch.tensor(stop)), 1, num_gaussians)  # Determines the center of each function
        self.beta = torch.pow(2*torch.pow(torch.tensor(num_gaussians), -1.)*(1 - torch.exp(-torch.tensor(stop))), -2.)  # Determines the width of each function
        self.register_buffer('offset', offset)

    def forward(self, dist):
        return torch.exp(-self.beta * torch.pow(torch.exp(-dist.view(-1, 1)) - self.offset.view(1, -1), 2))
    
    
############################################################################################################################
############################################################################################################################
# Define SchNet interaction block and cfconv layer (from PyG code)
############################################################################################################################
############################################################################################################################

# Define the SchNet interaction layer class
class InteractionBlock(torch.nn.Module):
    def __init__(self,
                 hidden_channels,
                 num_gaussians,
                 num_filters,
                 act_fn,
                 cutoff,
                 cosine_offset
                ):
        
        super().__init__()
        
        # Filter-generating NN
        self.filter_nn = torch.nn.Sequential(
            torch.nn.Linear(num_gaussians, num_filters),
            act_fn,
            torch.nn.Linear(num_filters, num_filters),
        )
        
        # Define continuous-filter convolution layer
        self.conv = CFConv(hidden_channels,
                           hidden_channels,
                           num_filters,
                           self.filter_nn,
                           cutoff,
                           cosine_offset
        )
        
        # Define other layers
        self.act = act_fn
        self.linear3 = torch.nn.Linear(hidden_channels, hidden_channels)
        
        # Reset parameters of network
        self.reset_parameters()
        
    def reset_parameters(self):
        # Filter NN layer 1 reset weights/biases
        torch.nn.init.xavier_uniform_(self.filter_nn[0].weight)
        self.filter_nn[0].bias.data.fill_(0)
        
        # Filter NN layer 3 reset weights/biases
        torch.nn.init.xavier_uniform_(self.filter_nn[2].weight)
        self.filter_nn[2].bias.data.fill_(0)
        
        # Reset CFConv parameters
        self.conv.reset_parameters()
        
        # Reset Linear3 parameters
        torch.nn.init.xavier_uniform_(self.linear3.weight)
        self.linear3.bias.data.fill_(0)
        
    def forward(self,
                x,
                ji_pairs,
                e_ji,
                e_ji_basis
               ):
        
        x = self.conv(x, ji_pairs, e_ji, e_ji_basis)
        x = self.act(x)
        x = self.linear3(x)
        return x
    
    
# Define the SchNet CFConv layer class
class CFConv(MessagePassing):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 num_filters,
                 filter_nn,   # filter-generating network to calculate W
                 cutoff,
                 cosine_offset
                ):
    
        # Pass aggr parameter (aggregate message from neighbors by addition)
        super().__init__(aggr='add')
        
        # Set input parameters
        self.filter_nn = filter_nn
        self.cutoff = cutoff
        self.cosine_offset = cosine_offset
    
        # Build NN layers
        self.linear1 = torch.nn.Linear(in_channels, num_filters, bias=False)
        self.linear2 = torch.nn.Linear(num_filters, out_channels)
    
        # Reset parameters of network
        self.reset_parameters()
    
    def reset_parameters(self):
        # Reset weights of each layer
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        
        # Reset bias
        self.linear2.bias.data.fill_(0)
            
    # Define cosine cutoff
    def cosine_cutoff(self, e_ji):
        # Modified cosine cutoff (shifted, scaled between 0.5 to 0)
        C = 0.25 * (torch.cos((e_ji - self.cosine_offset)\
                    * torch.tensor(math.pi) / (self.cutoff - self.cosine_offset)) + 1.0)
        return C
    
    # Message to propagate to nearby nodes
    def message(self, x_j, W):
        return x_j * W
    
    def forward(self,
                x,
                ji_pairs,
                e_ji,
                e_ji_basis
               ):
        
        
        # Calculate Behler cosine cutoff to scale filter
        C = self.cosine_cutoff(e_ji)

        # Generate filter with filter_nn, apply cutoff to filters
        W = self.filter_nn(e_ji_basis) * C.view(-1, 1)  # 1D reshape, stacks elements in order of appearance
        
        # Pass message
        x = self.linear1(x)
        x = self.propagate(ji_pairs, x=x, W=W)  # calc message with x, W parms
        x = self.linear2(x)
        return x
        

############################################################################################################################
############################################################################################################################
# Define function to create SchNet layers (Note, the SAKE function will handle the energy NN, embedding, etc)
############################################################################################################################
############################################################################################################################
    
def create_SchNet_layers(hidden_channels,
                         num_filters,
                         num_interactions,
                         num_gaussians, 
                         cutoff,
                         cosine_offset,
                         act_fn
                        ):
    
    ## Create interaction block(s)
    
    # Create module list
    interactions = torch.nn.ModuleList()
    
    # Add num_interactions interaction blocks to self.interactions ModuleList
    for _ in range(num_interactions):
        block = InteractionBlock(hidden_channels,
                                 num_gaussians,
                                 num_filters,
                                 act_fn,
                                 cutoff,
                                 cosine_offset
                                )
        
        interactions.append(block)
    
    # Define RBF function for SchNet
    rbf_func = expnorm_smearing(stop=cutoff, num_gaussians=num_gaussians)
    
    return interactions, rbf_func


#######################################################################################################################################
##############################               Define SAKE Layer from Wang & Chodera, 2023.                ##############################
#######################################################################################################################################
class SAKELayer(torch.nn.Module):
    """
    SAKE Layer, implemented based on code from
    E(n) Equivariant Convolutional Layer
    re
    """

    def __init__(self, 
                 input_nf, 
                 output_nf, 
                 hidden_nf, 
                 edges_in_d=0,
                 act_fn=torch.nn.CELU(alpha=2.0), 
                 n_heads=4, 
                 cutoff=0.5,
                 kernel_size=18, 
                ):
        
        super(SAKELayer, self).__init__()
        input_edge = input_nf * 2
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
        self.semantic_att_mlp = torch.nn.Sequential(torch.nn.Linear(hidden_nf, self.n_heads),
                                                    torch.nn.CELU(alpha=2.0),
                                                    torch.nn.Linear(self.n_heads, 1)
                                                   )

        # Radial basis function, projection
        self.rbf_model = torch.nn.Linear(kernel_size, hidden_nf)
        
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
    def edge_model(self, source, target, radial, rbf):
        # Project RBF to hidden_nf dimensions
        rbf = self.rbf_model(rbf)
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
        C = 0.5 * (torch.cos(dist * torch.tensor(math.pi) / 2*self.cutoff) + 1.0)
        return C
    
    # Define distance and semantic attention
    def dist_x_semantic_attn(self, radial, edge_attr):
        # Distance-based attention
        euclidean_att = self.cosine_cutoff(radial.sqrt())
        
	    # Semantic attention
        semantic_att = self.semantic_att_mlp(edge_attr) # Output same shape as edge embedding, perform element-wise mult w/ edges
        return semantic_att * euclidean_att

    
    # Node featurization
    def node_model(self, x, edge_index, edge_attr, spatial_att):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        agg = torch.cat([x, agg, spatial_att], dim=1)
        out = self.node_mlp(agg)
        return out
    

    # Forward function for SAKE layer
    def forward(self, h, edge_index, radial, coord_diff, rbf):
        row, col = edge_index

        edge_feat = self.edge_model(h[row], h[col], radial, rbf)
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
                       cutoff=0.5,
                       embed_type = 'c36',
                       energy_NN_layers = 3,
                       kernel_size = 18
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
                         act_fn=act_fn, n_heads=n_heads, cutoff=cutoff, kernel_size=kernel_size
                        )
        
        sake_conv.append(conv)
        
    # Create energy-predicting NN
    if energy_NN_layers == 3:
        energy_network = torch.nn.Sequential(torch.nn.Linear(out_node_nf, out_node_nf//2),
                                             energy_act_fn,
                                             torch.nn.Linear(out_node_nf//2, out_node_nf//4),
                                             energy_act_fn,
                                             torch.nn.Linear(out_node_nf//4, 1)
                                            )
        
    if energy_NN_layers == 2:
        energy_network = torch.nn.Sequential(torch.nn.Linear(out_node_nf, out_node_nf//2),
                                             energy_act_fn,
                                             torch.nn.Linear(out_node_nf//2, 1)
                                            )
    
    # Define RBF function for SAKE
    rbf_func = expnorm_smearing(stop=2*cutoff**2, num_gaussians=kernel_size)
    
    # Return all layers
    return embedding_in, embedding_out, sake_conv, energy_network, rbf_func
    

##############################################################################################################################
### Create Schake model
##############################################################################################################################
def create_Schake(hidden_channels, num_layers, kernel_size, cosine_offset,
                  sake_low_cut, sake_high_cut, schnet_low_cut, schnet_high_cut, 
                  schnet_act, sake_act, out_act, normalize, max_num_neigh, schnet_sel,
                  num_heads, embed_type, num_out_layers, device):
    
    # Create SchNet layers
    schnet_blocks, schnet_rbf = create_SchNet_layers(hidden_channels = hidden_channels,
                                                     num_filters = hidden_channels,
                                                     num_interactions = num_layers,
                                                     num_gaussians = kernel_size,
                                                     cutoff = schnet_high_cut,
                                                     cosine_offset = cosine_offset,
                                                     act_fn = schnet_act
                                                    )
        
    # Create SAKE layers
    embed_in, embed_out, sake_blocks, \
    energy_NN, sake_rbf = create_SAKE_layers(in_node_nf = 1,
                                             hidden_nf = hidden_channels, 
                                             out_node_nf = hidden_channels, 
                                             act_fn = sake_act, 
                                             energy_act_fn = out_act, 
                                             n_layers = num_layers, 
                                             n_heads = num_heads, 
                                             cutoff = sake_high_cut,
                                             embed_type = embed_type, 
                                             energy_NN_layers = num_out_layers,
                                             kernel_size = kernel_size
                                            )
    
    # Create Schake model
    model = Schake_modular(embedding_in = embed_in,
                           embedding_out = embed_out,
                           sake_rbf_func = sake_rbf,
                           schnet_rbf_func = schnet_rbf,
                           sake_layers = sake_blocks,
                           schnet_layers = schnet_blocks,
                           energy_network = energy_NN,
                           sake_low_cut = sake_low_cut,
                           sake_high_cut = sake_high_cut,
                           schnet_low_cut = schnet_low_cut,
                           schnet_high_cut = schnet_high_cut,
                           max_num_neigh = max_num_neigh,
                           normalize = normalize,
                           h_schnet = schnet_sel,
                           device = device
                          )
    
    return model
    
