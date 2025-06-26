import torch
import numpy as np

def hist_coord(edges):
  """Return the coordinate of bins from edges."""
  size = edges.shape[0]-1
  coord = torch.zeros(size)
  for i in range(size):
    coord[i] = (edges[i]+edges[i+1])/2
  return coord

def get_subsample_index(inputs,subsamples=1000):
            ind_list = np.linspace(0,len(inputs)-1,len(inputs)).astype(int)
            batch_inds = torch.tensor(np.array([np.random.choice(ind_list,subsamples)])).long().view(subsamples,)
            return batch_inds

def split_data_index(data,n_splits):
     ind_list = np.linspace(0,len(data)-1,len(data)).astype(int)
     np.random.shuffle(ind_list)
     return np.array_split(ind_list,n_splits)
