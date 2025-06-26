import torch
from torch.utils.data import Dataset

import numpy as np

from scipy.special import erf
from scipy.stats import laplace

from doubleml.datasets import fetch_401K

# Z-score data normalization

class StandardScaler:
    # Adapted from: https://gist.github.com/aryan-f/8a416f33a27d73a149f92ce4708beb40
    def __init__(self, mean=None, std=None, epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)
    
    def inv_transform(self,values):
        return values*(self.std + self.epsilon) + self.mean

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

def create_input(x,scaler=None):
   if scaler:
       x = scaler.transform(x)
   u = torch.normal(mean=0,std=1,size=(x.shape[0],1))
   return torch.hstack([u,x]).detach()

# Toy dataset

def mean_signal(x):
   return 10*np.sin(np.pi*x/14.0)

def sample_laplace(loc=0,scale=1,size=(1,1)):
    """Return a random sample from the Laplace distribution."""
    x = np.random.laplace(loc, scale, size)
    return torch.tensor(x, dtype=torch.float32,requires_grad=False)

def normal_to_laplace(u):
  """Transport map from the normal distribution to the Laplace distribution."""
  u = 0.5+0.5*erf(u/np.sqrt(2))
  u = laplace.ppf(u)
  return u

class ToyDataSet(Dataset):
  """Toy dataset where P(x)=Unif([0,10]) and P(y|x) = m(x) + noise."""
  def __init__(self, n_samples, noise="Laplace") -> None:

    x = 10.0*torch.rand(size=(n_samples, 1))
    if noise=='Gauss':
       y = 10.0*torch.sin(x*np.pi/14.0) + torch.normal(mean=0, std=1, size=(n_samples, 1))
    elif noise=='Laplace':
       y = 10.0*torch.sin(x*np.pi/14.0) + sample_laplace(0, 1, (n_samples,1))
    else:
       raise Exception("Noise must be either Gauss or Laplace")
         
    self.x = x
    self.y = y

    self.n_samples = n_samples

  def __getitem__(self, index):
      return self.x[index], self.y[index]

  def __len__(self):
      return self.n_samples

# Real dataset

class Fetch_401k_DataSet(Dataset):

    def __init__(self) -> None:

        Data = fetch_401K(return_type='DataFrame')
        covariates = ['age', 'inc', 'educ', 'fsize', 'marr','twoearn', 'db', 'pira', 'hown', 'e401']
        treatment = ["e401"]
        outcome = ["net_tfa"]
        X = Data[Data.columns.intersection(covariates)]
        names_x = np.array(list(X[:0]))
        treatment_ind = np.where(names_x == "e401")[0][0]
        X = X.to_numpy()
        n = len(X)
        d = len(X.T)

        cols_order = ([treatment_ind]+
                        list(np.linspace(0,treatment_ind-1,treatment_ind).astype(int))+
                        list(np.linspace(treatment_ind+1,d-1,d-1-treatment_ind).astype(int)))
        X = X[:,cols_order]
        Y = Data[Data.columns.intersection(outcome)].to_numpy().reshape(n,)
        X,Y = torch.tensor(X),torch.tensor(Y).view(n,1)

        shuffled_inds = torch.randperm(n)
        X = X[shuffled_inds]
        Y = Y[shuffled_inds]

        self.x1 = X
        self.x2 = Y

        self.n_samples = n
        self.n_features = d+1

    def __getitem__(self, index):
        return self.x1[index], self.x2[index]

    def __len__(self):
        return self.n_samples
        
    def shape(self):
        return (self.n_samples, self.n_features)