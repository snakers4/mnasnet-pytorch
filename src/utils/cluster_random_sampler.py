import random
from torch.utils.data.sampler import Sampler

class ClusterRandomSampler(Sampler):
    r"""Takes a dataset with cluster_indices property, cuts it into batch-sized chunks
    Drops the extra items, not fitting into exact batches
    Arguments:
        data_source (Dataset): a Dataset to sample from. Should have a cluster_indices property
        batch_size (int): a batch size that you would like to use later with Dataloader class
        shuffle (bool): whether to shuffle the data or not
    """

    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle

        batch_lists = []
        for j, cluster_indices in enumerate(self.data_source.cluster_indices):
            
            if hasattr(self.data_source, 'oversampling_indices'):
                print('Oversampling initiated')
                cluster_oversampling_indices = self.data_source.oversampling_indices[j]
                assert len(cluster_oversampling_indices) == len(cluster_indices)
                cluster_indices = [[_]*cluster_oversampling_indices[k] for k,_ in enumerate(cluster_indices)]
                cluster_indices = self.flatten_list(cluster_indices)
                # shuffle so that we do not have the same image dominating one batch
                if self.shuffle:
                    random.shuffle(cluster_indices)
                
            batches = [cluster_indices[i:i + self.batch_size] for i in range(0, len(cluster_indices), self.batch_size)]
            # filter our the shorter batches
            batches = [_ for _ in batches if len(_) == self.batch_size]
            if self.shuffle:
                random.shuffle(batches)
            batch_lists.append(batches)       
        
        # flatten lists and shuffle the batches if necessary
        # this works on batch level
        lst = self.flatten_list(batch_lists)
        if self.shuffle:
            random.shuffle(lst)

        self.lst = lst
        
    def flatten_list(self, lst):
        return [item for sublist in lst for item in sublist]

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.lst)
        return iter(self.flatten_list(self.lst))

    def __len__(self):
        return len(self.flatten_list(self.lst))