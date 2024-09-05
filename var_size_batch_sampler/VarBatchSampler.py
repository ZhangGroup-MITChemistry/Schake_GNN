import torch
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union

# Define wrapper function for variable batch sampler
'''
Purpose of this function is to ensure that there are the same
number of batches in all of the variable batch samplers. This can
be added to training code to ensure synchronization during DDP 
training.
'''
class SamplerRepeats(torch.utils.data.sampler.Sampler):
    def __init__(self, batch_sampler, tot_num_batches):
        self.batch_sampler = batch_sampler
        self.tot_num_batches = tot_num_batches
        
    def __iter__(self) -> Iterator[List[int]]:
        # Convert batch sampler to list
        sampler_list = list(self.batch_sampler)
                
        # Loop through sampler list, yield all batches
        for i, batch in enumerate(sampler_list):
            yield batch
        
        # Get random batches to repeat if necessary
        if i < self.tot_num_batches:
            # Get random permutations of idxs in list, only get necessary idxs
            random_batches = torch.randperm(i)[:self.tot_num_batches-(i+1)]
            assert len(random_batches) + (i+1) == self.tot_num_batches
            # Loop through random batches, yield all repeats
            for repeat in random_batches:
                yield sampler_list[repeat]
    
    def __len__(self) -> int:
        return self.tot_num_batches


# Define variable batch sampler
class VarBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, sampler, max_batch_size, n_atom_limit, dataset, drop_last=False):
        
        # Set variables to inherit
        self.sampler = sampler
        self.max_batch_size = max_batch_size
        self.n_atom_limit = n_atom_limit
        self.dataset = dataset
        self.drop_last = drop_last
        
        # Initialize batch_count
        self.batch_count = 0
        
        ## Compute batch count (i.e., length sampler, 
        ## essentially do the same thing as in __inter__
 
        # Convert original sampler into a list
        list_sampler = list(self.sampler)
        
        # Initialize index in count, n_atom count
        idx_in_batch, n_atom_count = 0, 0
        
        # Loop through the sampler
        for idx in range(1,len(list_sampler)):

            # Iterate idx_in_batch and n_atom_count
            idx_in_batch += 1
            n_atom_count += self.dataset[list_sampler[idx-1]][-2]  # Get dataset size
            
            # If adding the next sample would put batch over n_atom_limit and not at final idx,
            # iterate the batch_count
            if (n_atom_count + self.dataset[list_sampler[idx]][-2] > self.n_atom_limit):
                if (idx != len(list_sampler) - 1):
                    idx_in_batch, n_atom_count = 0, 0
                    self.batch_count += 1
                
                
                # If at final idx and over atom limit
                else:
                    # Iterate batch count for current batch 
                    self.batch_count += 1
                    # Iterate batch count for final idx that would be over n_atom_limit
                    self.batch_count += 1
                
                    
            # Only check after confirming not over n_atom_limit and not at final idx
            # If batch_size met, iterate the batch_count
            if idx_in_batch == self.max_batch_size:
                idx_in_batch, n_atom_count = 0, 0
                self.batch_count += 1
            
            # Only performed after confirming under max_batch_limit
            # If adding next sample would NOT put batch over n_atom_limit and next sample
            # is the final idx, add next sample to batch and return this final batch
            if (n_atom_count + self.dataset[list_sampler[idx]][-2] <= self.n_atom_limit)\
            and (idx == len(list_sampler) - 1):
                self.batch_count += 1
            
        
    def __iter__(self) -> Iterator[List[int]]:
        '''
        Note, we do not include a drop_last option here. This is because the batch
        sizses are inherently unequal using this varaible batch loader. As such, a
        drop_last option doesn't make sense.
        '''
        
        # Initialize idx_in_batch, n_atom_count
        batch, idx_in_batch, n_atom_count = [], 0, 0
        
        # Convert the sampler into a list
        list_sampler = list(self.sampler)
        
        # Loop through the sampler
        '''
        Note, the range is 1,len(list_sampler) because we want to be 1 sample idx
        ahead of the loop when adding samples from element list_sampler[idx-1] to
        the batch. This is so that if the next sample would put us over the atom
        limit, the batch is yielded as is, and that next sample then goes into
        the next batch. This is why we want to be at idx+1 and not idx in loop.
        '''
        for idx in range(1,len(list_sampler)):
            # Append idx to batch list
            batch.append(list_sampler[idx-1])

            # Iterate idx_in_batch and n_atom_count
            idx_in_batch += 1
            n_atom_count += self.dataset[list_sampler[idx-1]][-2]
            
            # If adding the next sample would put batch over n_atom_limit and not at final idx,
            # return the batch. This will put next sample in next batch.
            if (n_atom_count + self.dataset[list_sampler[idx]][-2] > self.n_atom_limit):
                if (idx != len(list_sampler) - 1):
                    yield batch
                    batch, idx_in_batch, n_atom_count = [], 0, 0
                
                # If at final idx and over atom limit on next addition
                else:
                    # Return current batch
                    yield batch
                    # Return final idx that would put over n_atom_limit as own batch
                    yield [idx]
                    
            # Only check after confirming not over n_atom_limit and not at final idx
            # If batch_size met, end batch
            if idx_in_batch == self.max_batch_size:
                yield batch
                batch, idx_in_batch, n_atom_count = [], 0, 0
            
            # Only performed after confirming under max_batch_limit
            # If adding next sample would NOT put batch over n_atom_limit and next sample
            # is the final idx, add next sample to batch and return this final batch
            if (n_atom_count + self.dataset[list_sampler[idx]][-2] <= self.n_atom_limit)\
            and (idx == len(list_sampler) - 1):
                batch.append(list_sampler[idx])
                yield batch
            

    def __len__(self) -> int:
        return self.batch_count
