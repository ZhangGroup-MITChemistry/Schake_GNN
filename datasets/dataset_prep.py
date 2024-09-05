# Imports
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn import preprocessing
import os
import warnings

# Define CoordsDataset object for data
class CoordsDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 coords, 
                 energies, 
                 forces, 
                 Z, 
                 pdb_names, 
                 n_atoms,
                 pdb_filename
                ):
        
        # Assert same shape of all data for molecule 1, block construction if not same shape
        assert coords[0].shape == forces[0].shape
        assert coords[0].shape[0] == Z[0].shape[0]
        assert coords[0].shape[0] == pdb_names[0].shape[0]
        assert coords[0].shape[0] == n_atoms[0]
        
        # Set variables
        self.coords = coords
        self.energies = energies
        self.forces = forces
        self.Z = Z
        self.pdb_names = pdb_names
        self.n_atoms = n_atoms
        self.pdb_filename = pdb_filename
        
    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, idx):
        
        return torch.Tensor(self.coords[idx]), torch.Tensor([self.energies[idx]]), torch.Tensor(self.forces[idx]),\
               torch.LongTensor(self.Z[idx]), torch.LongTensor(self.pdb_names[idx]),\
               torch.LongTensor([self.n_atoms[idx]]), self.pdb_filename[idx]
    
    
# Define CoordsDataset object for data
class CoordsDataset_no_forces(torch.utils.data.Dataset):
    def __init__(self, 
                 coords, 
                 energies, 
                 #forces, 
                 Z, 
                 pdb_names, 
                 n_atoms,
                 pdb_filename
                ):
        
        # Assert same shape of all data for molecule 1, block construction if not same shape
        #assert coords[0].shape == forces[0].shape
        assert coords[0].shape[0] == Z[0].shape[0]
        assert coords[0].shape[0] == pdb_names[0].shape[0]
        assert coords[0].shape[0] == n_atoms[0]
        
        # Set variables
        self.coords = coords
        self.energies = energies
        #self.forces = forces
        self.Z = Z
        self.pdb_names = pdb_names
        self.n_atoms = n_atoms
        self.pdb_filename = pdb_filename
        
    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, idx):
        
        return torch.Tensor(self.coords[idx]), torch.Tensor([self.energies[idx]]),\
               torch.LongTensor(self.Z[idx]), torch.LongTensor(self.pdb_names[idx]),\
               torch.LongTensor([self.n_atoms[idx]]), self.pdb_filename[idx]
    
    
# Define embedding from sklearn for PDB names and Z
class get_embedding(object):
    def __init__(self, embed_type: str):
        # Set embed type
        self.embed_type = embed_type
        
        # Set labels depending on type
        if (self.embed_type == 'element') or (self.embed_type == 'elements'):
            labels = ['H', 'C', 'N', 'O', 'S', 'F', 'Cl', 'I', 'Br']
            
        elif (self.embed_type == 'name') or (self.embed_type == 'names'):
            labels = ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 
                      'CG', 'CG1', 'CG2', 'CH2', 'CZ', 'CZ2', 'CZ3', 'H', 'H1', 'H2', 
                      'H3', 'HA', 'HA2', 'HA3', 'HB', 'HB1', 'HB2', 'HB3', 'HD1', 'HD11', 
                      'HD12', 'HD13', 'HD2', 'HD21', 'HD22', 'HD23', 'HD3', 'HE', 'HE1', 'HE2',
                      'HE21', 'HE22', 'HE3', 'HG', 'HG1', 'HG11', 'HG12', 'HG13', 'HG2', 'HG21',
                      'HG22', 'HG23', 'HG3', 'HH', 'HH11', 'HH12', 'HH2', 'HH21', 'HH22', 'HZ',
                      'HZ1', 'HZ2', 'HZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 
                      'NH2', 'NZ', 'O', 'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 
                      'OXT', 'SD', 'SG']
            
        else:
            raise ValueError('Unsupported embed_type: {}'.format(self.embed_type))
        
        # Initialize sklearn label encoder
        self.embed = preprocessing.LabelEncoder()
        self.embed.fit(labels)
        
    def __call__(self, features):
        # Get the label encoding
        return self.embed.transform(features)


##################################################################################################################
# #################################################################################################################
# Define collation function for batch training
class collation_func(object):
    def __init__(self, embed_type: str, unit: str, use_forces: bool = False, return_entry: bool = False):
                 
        # Conver to self variables
        self.embed_type = embed_type
        self.unit = unit
        self.use_forces = use_forces
        self.return_entry = return_entry
                                    
        # Raise errors if incorrect embed_type
        if self.embed_type not in ['elements', 'names']:
            raise ValueError("Embed type must be 'charges' or 'names'. '{}' is unsupported".format(self.embed_type))
        if self.unit not in ['kcal/mol', 'kJ/mol']:
            raise ValueError("Unit must be either 'kcal/mol' or 'kJ/mol'. {} is unsupported".format(self.unit))

        # Set unit conversion factor
        if unit == 'kJ/mol':
            self.conversion = 1     
        elif unit == 'kcal/mol':
            self.conversion = 4.184
        else:
            raise ValueError("Unsupported units.")
                                  
    def __call__(self, batch):
	
        # Create lists to append data to
        coords_batch = []
        energies_batch = []

        if self.use_forces:
            forces_batch = []

        if self.embed_type == 'elements':
            elements_batch = []
        if self.embed_type == 'names':
            names_batch = []

        n_atoms_batch = []

        if self.return_entry:
            entry_batch = []

        batch_tensor = []

        for i, b in enumerate(batch):
            coord, energy, force, element, name, n_atom, entry = b

            # Convert units
            energy = energy / self.conversion

            # Create batch tensor for each structure
            batch = (torch.zeros(n_atom)+i).type(torch.LongTensor)

            # Append all data to lists
            coords_batch.append(coord)
            energies_batch.append(energy)

            if self.use_forces:
                force = force / self.conversion
                forces_batch.append(force)

            if self.embed_type == 'elements':
                elements_batch.append(element)
            if self.embed_type == 'names':
                names_batch.append(name)

            n_atoms_batch.append(n_atom)

            if self.return_entry:
                entry_batch.append(entry)

            batch_tensor.append(batch)

        # Flatten all lists, return
        coords_batch = torch.cat(coords_batch)
        energies_batch = torch.cat(energies_batch)

        if self.use_forces:
            forces_batch = torch.cat(forces_batch)

        if self.embed_type == 'elements':
            elements_batch = torch.cat(elements_batch)
        if self.embed_type == 'names':
            names_batch = torch.cat(names_batch)

        n_atoms_batch = torch.cat(n_atoms_batch)
        batch_tensor = torch.cat(batch_tensor)

        # Return all necessary data
        if (self.use_forces == True) and (self.return_entry == True) and (self.embed_type == 'elements'):
            return coords_batch, energies_batch, forces_batch, elements_batch, \
                   n_atoms_batch, batch_tensor, entry_batch

        if (self.use_forces == True) and (self.return_entry == False) and (self.embed_type == 'elements'):
            return coords_batch, energies_batch, forces_batch, elements_batch, \
                   n_atoms_batch, batch_tensor

        if (self.use_forces == False) and (self.return_entry == True) and (self.embed_type == 'elements'):
            return coords_batch, energies_batch, elements_batch, \
                   n_atoms_batch, batch_tensor, entry_batch

        if (self.use_forces == False) and (self.return_entry == False) and (self.embed_type == 'elements'):
            return coords_batch, energies_batch, elements_batch, \
                   n_atoms_batch, batch_tensor

        if (self.use_forces == True) and (self.return_entry == True) and (self.embed_type == 'names'):
            return coords_batch, energies_batch, forces_batch, names_batch, \
                   n_atoms_batch, batch_tensor, entry_batch

        if (self.use_forces == True) and (self.return_entry == False) and (self.embed_type == 'names'):
            return coords_batch, energies_batch, forces_batch, names_batch, \
                   n_atoms_batch, batch_tensor

        if (self.use_forces == False) and (self.return_entry == True) and (self.embed_type == 'names'):
            return coords_batch, energies_batch, names_batch, \
                   n_atoms_batch, batch_tensor, entry_batch

        if (self.use_forces == False) and (self.return_entry == False) and (self.embed_type == 'names'):
            return coords_batch, energies_batch, names_batch, \
                   n_atoms_batch, batch_tensor


