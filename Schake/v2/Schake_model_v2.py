# Imports
import torch
import numpy as np
import math
from torch_scatter import scatter, scatter_add
from sklearn.preprocessing import OneHotEncoder
import mdtraj

# Import ML libraries
from torch.utils.data import DataLoader
from torch_geometric.data.makedirs import makedirs
from torch_geometric.nn import MessagePassing, radius_graph
import copy

# Other misc. libraries
import re
from collections import OrderedDict

############################################################################################################################
############################################################################################################################
# Define the Schake model (assuming elements used for embedding)
# Note, usage of this independently is meant for model training
############################################################################################################################
############################################################################################################################ 
class Schake_modular_Zs(torch.nn.Module):
    def __init__(self,
                 embedding_in,
                 neigh_embed: str,
                 embedding_out,
                 sake_rbf_func,
                 schnet_rbf_func,
                 sake_layers,
                 schnet_layers,
                 sake_low_cut,
                 sake_high_cut,
                 schnet_low_cut,
                 schnet_high_cut,
                 out_network,
                 max_num_neigh = 10000,
                 h_schnet = 1,
                 device = 'cpu',
                ):
        
        super(Schake_modular_Zs, self).__init__()
        self.device = device
        self.max_num_neigh = max_num_neigh
        self.neigh_embed = neigh_embed
        
        # Define layers
        self.embedding_in = embedding_in
        self.embedding_out = embedding_out
        self.sake_rbf_func = sake_rbf_func
        self.schnet_rbf_func = schnet_rbf_func
        self.sake_layers = sake_layers
        self.schnet_layers = schnet_layers
        self.out_network = out_network      
        
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

        return radial, coord_diff
    
    # Define function to get edges, distances, radial, etc
    # (to be used in forward function, makes code more clear)
    def get_flt_edges(self, h, x, batch):
        # Generate adjacency lists
        edges = radius_graph(x, 
                             r=self.schnet_high_cut,
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
        # Note, h only used for filtering for Schake
        if self.h_schnet != None:
            
            # For SchNet pairs, create adjacency list in terms of species
            h_schnet_edges = h[schnet_edges]
            
            # Filter SchNet edges to only include atom type of interest
            h_mask = torch.where(h_schnet_edges[0] == self.h_schnet)[0]
            schnet_edges = schnet_edges.T[h_mask].T

            # Extract distance for SchNet pairs only
            schnet_dist = schnet_dist[h_mask]
            
        return sake_edges, schnet_edges, sake_radial, sake_coord_diff, schnet_dist
    
    # Define forward function
    def forward(self, h, z, x, batch):
        # Move necessary things to the device of choice
        x = x.to(self.device)
        batch = batch.to(self.device)
        h = h.to(self.device) # used for Schake layer, nothing else
        z = z.to(self.device)   # model acts on z here, not h
        
        # Filter edges, get coord_diff, radial, coord_diff
        sake_edges, schnet_edges,\
        sake_radial, sake_coord_diff, schnet_dist = self.get_flt_edges(h, x, batch)
        
        # Create radial basis functions
        sake_rbf, schnet_rbf = self.sake_rbf_func(sake_radial), self.schnet_rbf_func(schnet_dist)
        
        # Generate initial embedding, subset
        if self.neigh_embed == 'resid_bb3':
            z = torch.cat([self.embedding_in[0](z), self.embedding_in[1](h)], dim=-1) 
        else:
            z = self.embedding_in(z)
        
        # Run layers (SAKE for short-range, SchNet for long)
        for sake_int, schnet_int in zip(self.sake_layers, self.schnet_layers):
            z = z + sake_int(z, sake_edges, sake_radial, sake_coord_diff, sake_rbf, None)
            z = z + schnet_int(z, schnet_edges, schnet_dist, schnet_rbf, None)
            
        # Run embedding out layer
        z = self.embedding_out(z)
            
        # Run energy model
        z = self.out_network(z)

        return z
    
    
############################################################################################################################
############################################################################################################################
# Define the Schake model for single protein (energy outputting, only x inputted to forward function)
############################################################################################################################
############################################################################################################################ 
class Schake_modular_Zs_SP(torch.nn.Module):
    def __init__(self,
                 embedding_in,
                 neigh_embed: str,
                 embedding_out,
                 sake_rbf_func,
                 schnet_rbf_func,
                 sake_layers,
                 schnet_layers,
                 sake_low_cut,
                 sake_high_cut,
                 schnet_low_cut,
                 schnet_high_cut,
                 out_network,
                 max_num_neigh = 10000,
                 h_schnet = 1,
                 device = 'cpu',
                 return_logits = False,
                 energy_func = 'os',
                ):
        
        super(Schake_modular_Zs_SP, self).__init__()
        
        # Ensure valid energy model
        if energy_func not in ['os', 'ms']:
            raise ValueError(
                '{} not a valid energy function, must be "os" or "ms".'.format(energy_func))
        
        # Define pseudo_energy model
        self.pseudo_energy = Schake_modular_Zs(embedding_in, neigh_embed, embedding_out,
                                               sake_rbf_func, schnet_rbf_func, sake_layers,
                                               schnet_layers, sake_low_cut, sake_high_cut,
                                               schnet_low_cut, schnet_high_cut, out_network, 
                                               max_num_neigh, h_schnet, device
                                              )
        
        # Move other necessary variable to self
        self.device = device
        self.return_logits = return_logits
        
        # Set energy func value
        if energy_func == 'os':
            self.energy_func = 0
        elif energy_func == 'ms':
            self.energy_func = 1
        
        # Move model to device
        self.to(self.device)
    
    # Define function to load necessary self-inputs
    def load_mol_info(self, h, z, ohv, batch, #ang_idxs, dih_idxs, 
                      ca_idxs, temp = 1, epsilon = 1e-8,
                     ):
        
        # Ensure batch size is only one (this doesn't support multi-protein input)
        if batch.max() != 0:
            raise RuntimeError("Single-protein model only supports 1 configuration per batch.")
            
        # Compute kBT for energy calc
        kB = torch.tensor(1.3806490000000003e-26)  # kilojoules/kelvin
        NA = torch.tensor(6.02214076e+23) # /mol
        kBT = kB * torch.tensor(temp) * NA    
        
        # Define molecule-specific variables to pass to forward
        self.h, self.z, self.ohv = h.to(self.device), z.to(self.device), ohv.to(self.device)
        self.batch, self.ca_idxs = batch.to(self.device), ca_idxs.to(self.device)
        
        # Move variables necessary for energy calc
        self.kBT = kBT.to(self.device)
        self.epsilon = torch.tensor(epsilon).to(self.device)
    
    # Compute energies (assumes only 1 config per batch)
    def compute_energy_os(self, pseudo_e):
        prob = -self.kBT * torch.log(pseudo_e.softmax(dim=-1) + self.epsilon)
        energy = torch.sum(prob * self.ohv)
        return energy
    
    # Compute energies (assumes only 1 config per batch)
    def compute_energy_ms(self, pseudo_e):
        probs = pseudo_e.softmax(dim=-1) + self.epsilon
        max_prob = self.epsilon * torch.logsumexp(probs/self.epsilon, 1)
        return -self.kBT * torch.sum(torch.log(max_prob))
    
    # Forward pass
    def forward(self, x):
        # Move necessary things to the device of choice
        x = x.to(self.device)[self.ca_idxs]
        
        # Compute pseudo energy
        p = self.pseudo_energy(self.h, self.z, x, self.batch)
        
        # Compute energy
        if self.energy_func == 0:
            e = self.compute_energy_os(p)
        elif self.energy_func == 1:
            e = self.compute_energy_ms(p)
        
        if self.return_logits:
            return e, p
        else:
            return e

    
############################################################################################################################
############################################################################################################################
# Define expnorm smearing function (need to fix, doesn't work as intended)
############################################################################################################################
############################################################################################################################

class expnorm_smearing(torch.nn.Module):
    def __init__(self, cutoff_lower=0, cutoff_upper=5.0, num_gaussians=32, trainable=False):
        super().__init__()
        self.cutoff_upper = torch.tensor(cutoff_upper)
        self.cutoff_lower = torch.tensor(cutoff_lower)
        
        start = torch.exp(-self.cutoff_upper + self.cutoff_lower)
        offset = torch.linspace(start, 1, num_gaussians)  # Determines the center of each function
        beta = torch.pow(2*torch.pow(torch.tensor(num_gaussians), -1.)*(1 - start), -2.)  # Determines the width of each function
        
        if trainable:
            self.register_parameter('offset', torch.nn.Parameter(offset))
            self.register_parameter('beta', torch.nn.Parameter(beta))
        else:
            self.register_buffer('offset', offset)
            self.register_buffer('beta', beta)

    def forward(self, dist):
        return torch.exp(-self.beta * torch.pow(torch.exp(-dist.view(-1, 1)+self.cutoff_lower) - self.offset.view(1, -1), 2))
    
    
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
                 cosine_offset,
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
                           cosine_offset,
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
                e_ji_basis,
                mask = None
               ):
        
        x = self.conv(x, ji_pairs, e_ji, e_ji_basis, mask)
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
                 cosine_offset,
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
                e_ji_basis,
                mask = None
               ):
        
        
        # Calculate Behler cosine cutoff to scale filter
        C = self.cosine_cutoff(e_ji)

        # Generate filter with filter_nn, apply cutoff to filters
        W = self.filter_nn(e_ji_basis) * C.view(-1, 1)  # 1D reshape, stacks elements in order of appearance
        
        # Apply mask to neighboring embedding (note that W multiplied element-wise by x_j)
        if mask != None:
            W = W * mask
        
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
                         trainable_kernel,
                         cutoff_lower,
                         cutoff_upper,
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
                                 cutoff_upper,
                                 cutoff_lower,
                                )
        
        interactions.append(block)
    
    # Define RBF function for SchNet
    rbf_func = expnorm_smearing(cutoff_lower=cutoff_lower, 
                                cutoff_upper=cutoff_upper, 
                                num_gaussians=num_gaussians, 
                                trainable=trainable_kernel
                               )
    
    return interactions, rbf_func


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
                 act_fn=torch.nn.CELU(alpha=2.0), 
                 n_heads=4,
                 cutoff_lower=0,
                 cutoff_upper=0.5,
                 kernel_size=18, 
                ):
        
        super(SAKELayer, self).__init__()
        input_edge = input_nf * 2
        self.epsilon = 1e-8   # Add when dividing by parameters to prevent divide by 0
        edge_coords_nf = 1
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.n_heads = n_heads
        
        # Modefied for SAKE (hidden_nf*2)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(input_edge + edge_coords_nf + hidden_nf, hidden_nf),
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
    def edge_model(self, source, target, radial, rbf, mask=None):
        # Project RBF to hidden_nf dimensions
        rbf = self.rbf_model(rbf)
        # Apply masking to neighbor embedding (source)
        if mask != None:
            source = source * mask
        # Initialize edge embedding by concat node pair embeddings
        init_edge_feat = torch.cat([source, target], dim=1)
        # Pass concatenated node embeddings through filter nn
        W = self.filter_nn(init_edge_feat)
        # Concatenate all features for edge embedding
        out = torch.cat([init_edge_feat, radial, rbf*W], dim=1)
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
        '''
        Note, since we're filtering "row" in the forward function, scatter_add needs
        to act on "col" to correctly account for all nodes.
        '''
        # Note - 2nd input was originally row, changed to col for stability w/ filtering
        all_aggs = scatter_add(attn, col.unsqueeze(1), dim=0, dim_size=x.shape[0])
        
        # Input to mu MLP
        out = self.mu(torch.norm(all_aggs, dim=2))
        return out, all_aggs
    
    # Define cosine cutoff function
    def cosine_cutoff(self, dist, scalar_low):  # scalar_low controls end point (1 for 0, 2 for 0.5)
        #if not self.invert_cos_cut:
        C = 0.5 * (torch.cos((dist-self.cutoff_lower) * torch.tensor(math.pi)\
                             / (scalar_low*(self.cutoff_upper-self.cutoff_lower))) + 1.0)
        return C
    
    # Define distance and semantic attention
    def dist_x_semantic_attn(self, radial, edge_attr):
        # Distance-based attention
        euclidean_att = self.cosine_cutoff(radial.sqrt(), scalar_low=2)
        
	    # Semantic attention
        semantic_att = self.semantic_att_mlp(edge_attr) # Output same shape as edge embedding, perform element-wise mult w/ edges
        return semantic_att * euclidean_att

    
    # Node featurization
    def node_model(self, x, edge_index, edge_attr, spatial_att):
        row, col = edge_index
        '''
        Note, since we're filtering "row" in the forward function, unsorted_segment_sum 
        needs to act on "col" to correctly account for all nodes.
        '''
        agg = unsorted_segment_sum(edge_attr, col, num_segments=x.size(0))
        agg = torch.cat([x, agg, spatial_att], dim=1)
        out = self.node_mlp(agg)
        return out
    

    # Forward function for SAKE layer
    def forward(self, h, edge_index, radial, coord_diff, rbf, mask = None):
        row, col = edge_index

        edge_feat = self.edge_model(h[row], h[col], radial, rbf, mask)
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
                       embed_nf,
                       out_node_nf, 
                       act_fn=torch.nn.CELU(alpha=2.0), 
                       energy_act_fn=torch.nn.CELU(alpha=2.0), 
                       n_layers=4,
                       n_heads=4,
                       cutoff_lower=0,
                       cutoff_upper=0.5,
                       embed_type = 'c36',
                       energy_NN_layers = 3,
                       kernel_size = 18,
                       trainable_kernel = False
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
        num_types = 20
    elif embed_type == 'names':
        num_types = 83
    else:
        raise ValueError ('Invalid embedding type, must be "c36", "ff14SB", "gaff", "elements", or "names".')

    # Create embedding_in layer
    embedding_in = torch.nn.Embedding(num_types, embed_nf)

    # Create embedding_out layer
    embedding_out = torch.nn.LayerNorm(hidden_nf)
    
    # Create SAKE message passing layers
    sake_conv = torch.nn.ModuleList()
    for _ in range(n_layers):
        conv = SAKELayer(hidden_nf, hidden_nf, hidden_nf,
                         act_fn=act_fn, n_heads=n_heads, cutoff_lower=cutoff_lower,
                         cutoff_upper=cutoff_upper, kernel_size=kernel_size,
                        )
        
        sake_conv.append(conv)
        
    # Create energy-predicting NN
    if energy_NN_layers == 3:
        energy_network = torch.nn.Sequential(torch.nn.Linear(out_node_nf, 16),  # Output should be 20 for each amino acid
                                             energy_act_fn,
                                             torch.nn.Linear(16, 8),
                                             energy_act_fn,
                                             torch.nn.Linear(8, 8)
                                            )
        
    if energy_NN_layers == 2:
        energy_network = torch.nn.Sequential(torch.nn.Linear(out_node_nf, 16),
                                             energy_act_fn,
                                             torch.nn.Linear(16, 8)
                                            )
    
    # Define RBF function for SAKE
    rbf_func = expnorm_smearing(cutoff_lower=cutoff_lower**2, 
                                cutoff_upper=cutoff_upper**2, 
                                num_gaussians=kernel_size, 
                                trainable=trainable_kernel
                               )
    
    # Return all layers
    return embedding_in, embedding_out, sake_conv, energy_network, rbf_func
    
    

    
##############################################################################################################################
### Create Schake model (more light-weight model creation code)
##############################################################################################################################
def create_Schake(hidden_channels, num_layers, kernel_size, neighbor_embed: str,
                  sake_low_cut, sake_high_cut,
                  schnet_low_cut, schnet_high_cut, schnet_act, sake_act,
                  out_act, max_num_neigh, schnet_sel,
                  trainable_sake_kernel, trainable_schnet_kernel,
                  num_heads, embed_type, num_out_layers, device, 
                       single_pro = False, return_logits = False, energy_func = 'os'):    
    
    # Confirm valid embedding type
    valid_neigh_embed = ['resid', 'resid_bb3']
    if neighbor_embed not in valid_neigh_embed:
        raise RuntimeError('Invalid "neighbor_embed" option "{}" selected.'.format(neighbor_embed))
    
    # Set embedding width
    if (neighbor_embed == 'resid_bb3'):
        embed_width = hidden_channels // 2
    else:
        embed_width = hidden_channels
    
    # Create SchNet layers
    schnet_blocks, schnet_rbf = create_SchNet_layers(hidden_channels = hidden_channels,
                                                     num_filters = hidden_channels,
                                                     num_interactions = num_layers,
                                                     num_gaussians = kernel_size,
                                                     trainable_kernel = trainable_schnet_kernel,
                                                     cutoff_lower = schnet_low_cut,
                                                     cutoff_upper = schnet_high_cut,
                                                     act_fn = schnet_act,
                                                    )
        
    # Create SAKE layers
    embed_in, embed_out, sake_blocks, \
    energy_NN, sake_rbf = create_SAKE_layers(in_node_nf = 1,
                                             hidden_nf = hidden_channels, 
                                             embed_nf = embed_width,
                                             out_node_nf = hidden_channels, 
                                             act_fn = sake_act, 
                                             energy_act_fn = out_act, 
                                             n_layers = num_layers, 
                                             n_heads = num_heads, 
                                             cutoff_lower = sake_low_cut,
                                             cutoff_upper = sake_high_cut,
                                             embed_type = embed_type, 
                                             energy_NN_layers = num_out_layers,
                                             kernel_size = kernel_size,
                                             trainable_kernel = trainable_sake_kernel
                                            )
    

    if (neighbor_embed == 'resid_bb3'):
        embed_in = torch.nn.ModuleList([embed_in, # residue
                                        torch.nn.Embedding(64, embed_width) # bb3 atom name
                                       ])
    
    # Create Schake model
    if not single_pro:
        model = Schake_modular_Zs(embedding_in = embed_in,
                               neigh_embed = neighbor_embed,
                               embedding_out = embed_out,
                               sake_rbf_func = sake_rbf,
                               schnet_rbf_func = schnet_rbf,
                               sake_layers = sake_blocks,
                               schnet_layers = schnet_blocks,
                               out_network = energy_NN,
                               sake_low_cut = sake_low_cut,
                               sake_high_cut = sake_high_cut,
                               schnet_low_cut = schnet_low_cut,
                               schnet_high_cut = schnet_high_cut,
                               max_num_neigh = max_num_neigh,
                               h_schnet = schnet_sel,
                               device = device
                              )
        
    else:
        model = Schake_modular_Zs_SP(embedding_in = embed_in,
                               neigh_embed = neighbor_embed,
                               embedding_out = embed_out,
                               sake_rbf_func = sake_rbf,
                               schnet_rbf_func = schnet_rbf,
                               sake_layers = sake_blocks,
                               schnet_layers = schnet_blocks,
                               out_network = energy_NN,
                               sake_low_cut = sake_low_cut,
                               sake_high_cut = sake_high_cut,
                               schnet_low_cut = schnet_low_cut,
                               schnet_high_cut = schnet_high_cut,
                               max_num_neigh = max_num_neigh,
                               h_schnet = schnet_sel,
                               device = device,
                               return_logits = return_logits,
                               energy_func = energy_func
                              )
    
    return model

##############################################################################################################################
##############################################################################################################################
###################################################### HELPER FUNCTIONS ######################################################
##############################################################################################################################
##############################################################################################################################


##############################################################################################################################
### Define function to edit the state dict for single model
##############################################################################################################################
def _SP_state_dict(state_dict, single_pro=True):
    # Edit state dict to apply to this model
    new_dict = OrderedDict()
    for key, val in state_dict.items():
        if single_pro:
            new_dict['pseudo_energy.'+key] = val
        else:
            new_dict[key] = val
        
    return new_dict


##############################################################################################################################
### Define function convert 3-letter amino acids to 1-letter
##############################################################################################################################
def _3let_to_1let(_3let_codes):
    # Define dict to convert 3-letter codes to 1-letter codes
    _3let_to_1let = {'MET':'M', 'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D',
                     'CYS':'C', 'GLU':'E', 'GLN':'Q', 'GLY':'G', 'HIS':'H',
                     'ILE':'I', 'LEU':'L', 'LYS':'K', 'PHE':'F', 'PRO':'P',
                     'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V',
                    }
    
    # Convert 3-letter codes to 1-leter codes
    _1let_codes = np.array([_3let_to_1let[_3let] for _3let in _3let_codes])
    
    return _1let_codes


##############################################################################################################################
### Define function convert 1-letter amino acids to OHVs
##############################################################################################################################
def _1let_to_OHV(_1let_codes):
    # Define dict to convert 1-letter codes to OHVs
    _1let_to_OHV = {'M': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                    'A': np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                    'R': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
                    'N': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
                    'D': np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                    'C': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                    'E': np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                    'Q': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
                    'G': np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                    'H': np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                    'I': np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                    'L': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                    'K': np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                    'F': np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                    'P': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
                    'S': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
                    'T': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
                    'W': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
                    'Y': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
                    'V': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
                   }
    
    # Convert 1-letter codes to OHVs
    aa_ohvs = np.stack([_1let_to_OHV[_1let] for _1let in _1let_codes])
    
    return aa_ohvs


##############################################################################################################################
### Define function convert mdtraj DSSP data to OHVs
##############################################################################################################################
def _ss8_to_OHV(secstruct):
    # mapping dict
    map_dict = {'G' : np.array([1, 0, 0, 0, 0, 0, 0, 0]),
                'H' : np.array([0, 1, 0, 0, 0, 0, 0, 0]),
                'I' : np.array([0, 0, 1, 0, 0, 0, 0, 0]),
                'T' : np.array([0, 0, 0, 1, 0, 0, 0, 0]),
                'E' : np.array([0, 0, 0, 0, 1, 0, 0, 0]),
                'B' : np.array([0, 0, 0, 0, 0, 1, 0, 0]),
                'S' : np.array([0, 0, 0, 0, 0, 0, 1, 0]),
                'C' : np.array([0, 0, 0, 0, 0, 0, 0, 1]),
               }
    
    all_frame_ohe = []
    
    for j, frame in enumerate(secstruct):
        frame_ohe = []
        for i, char in enumerate(frame):
            frame_ohe.append(map_dict[char])
        all_frame_ohe.append(np.stack(frame_ohe))
    
    return np.stack(all_frame_ohe).squeeze()


##############################################################################################################################
### Define function to get data from an arbitrary PDB for inputting to model
##############################################################################################################################
def _load_mol(pdb_pathway, flt_grp = 'ca'):
    # Load PDB, top
    pdb = mdtraj.load_pdb(pdb_pathway)
    top = pdb.topology
    
    # Set valid flt_grps
    valid_flt_grp = ['ca', 'bb3']
    if flt_grp not in valid_flt_grp:
        raise RuntimeError('Invalid "flt_grp" selected: {}'.format(flt_grp))
    
    # Depending on flt_grp, get idxs
    if flt_grp == 'ca':
        sel_grp = ['CA']
    elif flt_grp == 'bb3':
        sel_grp = ['CA', 'C', 'N']
        
    # Set end caps, nonstandard residues to avoid
    avoid_res = ['ACE', 'NME', 'NHE', 'NLE']
    
    # Extract 3-letter codes for selection group atoms only
    _3let_codes = np.array([str(a.residue)[:3] for a in top.atoms\
                            if (a.name in sel_grp) and (str(a.residue)[:3] not in avoid_res)])
    
    # Extract selection group atoms only
    sel_idxs = np.array([i for i, a in enumerate(top.atoms)\
                         if (a.name in sel_grp) and (str(a.residue)[:3] in _3let_codes)])
    
    # Define dict to convert 3-letter codes to 1-letter codes
    _1let_codes = _3let_to_1let(_3let_codes)
    
    # Convert 1-letter codes to OHVs
    aa_ohvs = _1let_to_OHV(_1let_codes)
    
    # Convert OHVs to label embedding
    aa_labels = np.argmax(aa_ohvs, axis=-1)
    
    # Get PDB atom name embedding (1 for alpha carbons)
    if flt_grp == 'CA':
        atom_embed = np.ones_like(aa_labels)
    elif flt_grp == 'bb3':
        name_dict = {'CA':1, 'N':63, 'C':0}
        atom_embed = np.array([name_dict[a.name] for a in top.atoms\
                              if (a.name in sel_grp) and (str(a.residue)[:3] not in avoid_res)])
    
    # Compute DSSP labels
    ss8 = mdtraj.compute_dssp(pdb, False)
    ss8[ss8 == ' '] = 'C'
    ss8 = ss8[ss8 != 'NA']
    ss8_ohv = _ss8_to_OHV(ss8)
    if flt_grp == 'bb3':
        ss8_ohv = np.repeat(ss8_ohv, 3, axis=0)
    
    # Get coords of PDB
    coords = np.array(pdb.xyz[0])[sel_idxs]
    
    # Create batch tensor
    batch = np.zeros_like(atom_embed)
    
    # Return outputs
    return atom_embed, aa_labels, coords, batch, ss8_ohv, sel_idxs

