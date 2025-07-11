import torch

# Copied from: https://github.com/HWDance/Cocycles/blob/main/Package/causal_cocycle/kernels.py

class kernel:

    def __init__(self,lengthscale=None,scale=None):
        if lengthscale == None:
            self.lengthscale = torch.ones(1)
        else:
            self.lengthscale = lengthscale
        if scale ==None:
            self.scale = torch.ones(1)
        else:
            self.scale = scale
        
    def get_gram(self, X : torch.tensor, Z : torch.tensor):
        # Computes K_XZ
        raise NotImplementedError

class gaussian_kernel(kernel):
        
    def get_gram(self,X,Z):
        K_xx = torch.exp(-0.5*torch.cdist(X/self.lengthscale, Z/self.lengthscale, p=2.0)**2)            
        return K_xx*self.scale
    
class exponential_kernel(kernel):
        
    def get_gram(self,X,Z):
        K_xx = torch.exp(-0.5*torch.cdist(X/self.lengthscale, Z/self.lengthscale, p=2.0)**1)            
        return K_xx*self.scale
    
class inverse_gaussian_kernel(kernel):
        
    def get_gram(self,X,Z):
        K_xx = torch.exp(-0.5*torch.cdist(X*self.lengthscale, Z*self.lengthscale, p=2.0)**2)            
        return K_xx*self.scale
    

class multivariate_gaussian_kernel(kernel):
    
    def get_gram(self,X,Z):
        K_xx = torch.exp(-0.5*torch.cdist(X @ self.lengthscale, Z @ self.lengthscale, p=2.0)**2)
        return K_xx*self.scale

class linear_kernel(kernel):

    def get_gram(self,X,Z):
        return X @ Z.T
    
def median_heuristic(X):
    """
    Returns median heuristic lengthscale for Gaussian kernel
    """

    # Median heurstic for inputs
    Dist = torch.cdist(X,X, p = 2.0)**2
    Lower_tri = torch.tril(Dist, diagonal=-1).view(len(X)**2).sort(descending = True)[0]
    Lower_tri = Lower_tri[Lower_tri!=0]
    return (Lower_tri.median()/2).sqrt()