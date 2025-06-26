import torch
import copy
import numpy as np
import matplotlib.pyplot as plt

from utils.kernels import *
from utils.neural_networks import * 
from utils.data import *
from utils.basics import *
from torch.utils.data import DataLoader

# Distributional regression loss
# Adapted from: https://github.com/HWDance/Cocycles/blob/main/Package/causal_cocycle/loss_functions.py

class MMD_loss:
    
    def __init__(self,kernel):
        """
        kernel : kernel function
        """
        self.kernel = kernel
        
    def get_subsample(self,inputs,subsamples=[]):
            ind_list = np.linspace(0,len(inputs)-1,len(inputs)).astype(int)
            batch_inds = torch.tensor(np.array([np.random.choice(ind_list,subsamples)])).long().view(subsamples,)
            inputs_batch = inputs[batch_inds]
            return inputs_batch
                
    def median_heuristic(self,inputs,subsamples = 1000):
        """
        Returns median heuristic lengthscale for Gaussian kernel
        """
        
        # Subsampling
        if subsamples < len(inputs):
                inputs_batch = self.get_subsample(inputs,subsamples)
        else:
            inputs_batch = inputs
        
        # Median heurstic for inputs
        Dist = torch.cdist(inputs_batch,inputs_batch, p = 2.0)**2
        Lower_tri = torch.tril(Dist, diagonal=-1).view(len(inputs_batch)**2).sort(descending = True)[0]
        Lower_tri = Lower_tri[Lower_tri!=0]
        self.kernel.lengthscale =  (Lower_tri.median()/2).sqrt()
            
        return
    
    def loss_fn(self,model,x,y,n_samples=64):
        """
        model: conditional generator
        x : n_batch x d
        y : n_batch x 1
        """

        # Dimensions
        n_batch = len(y)
        y_hat = model.sample(x,n_samples)
        if len(y_hat.size())<3:
            y_hat = y_hat[...,None] # adding extra dimension to make sure output is N x N x 1 here

        # Computing kernels in batches to avoid memory overload
        K = 0
        if n_batch**3 >= 10**8:
            batchsize = max(1,min(n_batch,int(10**8/n_batch**2)))
        else:
            batchsize = n_batch
        n_subbatch = int(n_batch/batchsize)
        for i in range(n_subbatch):
            # Get gram matrices
            K += self.kernel.get_gram(y_hat[i*batchsize:(i+1)*batchsize],y_hat[i*batchsize:(i+1)*batchsize]).sum()/n_samples**2
            K += -2*self.kernel.get_gram(y[i*batchsize:(i+1)*batchsize,None,:],y_hat[i*batchsize:(i+1)*batchsize]).sum()/n_samples
        
        return K/n_batch

# Distributional regression model
class distributional_regression_model:

    def __init__(self,n_inputs,n_layers=1,hidden_size=128,monotonic=True,scalers=None,input_noise="Gaussian"):
        """
        Initialize a distributional regression model

        Parameters:
        - n_inputs: int, dimension of the x-input (without the noise input)
        - n_layers: int, number of hidden layers in the neural network.
        - hidden_size: int, dimension of the hidden layers in the neural network.
        - monotonic: boolean, whether the model is increasing in the noise input.
        - scalers: list, scalers to normalize the inputs and the outputs.
        - input_noise: "Gaussian" or "Uniform", distribution of the noise input.
        """
        if monotonic:
            monotonicity_indicator = torch.zeros((n_inputs+1,))
            monotonicity_indicator[0] = 1
            network = MonoNet(n_inputs=n_inputs+1,n_layers=n_layers,hidden_size=hidden_size,monotonicity_indicator=monotonicity_indicator)
        else:
            network = Net(n_inputs=n_inputs+1,layers=n_layers,hidden_size=hidden_size)
        self.network = network
        self.scalers = scalers
        self.input_noise = input_noise
    
    def det_forward(self,e,x,rescale=False):
        if self.scalers:
          x = self.scalers[0].transform(x)
        inputs = torch.hstack([e,x])
        outputs = self.network(inputs)
        if self.scalers and rescale:
          outputs = self.scalers[1].inv_transform(outputs)
        return outputs
    
    def sto_forward(self,x,rescale=False):
        if self.input_noise == "Gaussian":
          e = torch.normal(mean=0,std=1,size=(x.shape[0],1))
        else:
          e = torch.rand(size=(x.shape[0],1)) 
        if self.scalers:
          x = self.scalers[0].transform(x)
        inputs = torch.hstack([e,x])
        outputs = self.network(inputs)
        if self.scalers and rescale:
          outputs = self.scalers[1].inv_transform(outputs)
        return outputs
    
    def sample(self,x,n_sample,rescale=False):
        """
        Draw a sample from Y|X=x

        Inputs:
        - x: (n x d) torch.tensor, list of points at which samples are drawn
        - n_samples: int, number of samples to draw at each x-point

        Output:
        - y: (n x n_samples) torch.tensor 
        """
        # If scalers are stored, then automatically scales the input
        if self.scalers:
          x = self.scalers[0].transform(x)
        eye = torch.ones((n_sample,1))
        x_repeated = torch.kron(x,eye)
        if self.input_noise == "Gaussian":
          e = torch.normal(mean=0,std=1,size=(len(x_repeated),1))
        else:
          e = torch.rand(size=(len(x_repeated),1))
        inputs = torch.hstack([e,x_repeated])
        outputs = self.network(inputs)
        # If demanded, then rescales the output
        if self.scalers and rescale:
          outputs = self.scalers[1].inv_transform(outputs)
        return outputs.reshape(len(x),n_sample)
    
    def optimize(self,dataset,kernel=gaussian_kernel(),epochs=40,batch_size=64,lr=1e-3,scale=True,show=False):
        """
        Train the model on a datatset.

        Parameters:
        - dataset: DataSet, training data X,Y.
        - epochs: int, number of learning steps involving the whole training set.
        - batch_size: int, number of samples in the mini batches.
        - lr: float, learning rate.
        - scale: boolean, whether to train and store scalers for the model.
        - show: boolean, whether to show the learning curve
        """
        # Load train data
        dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=True)

        # Optimizer
        optimizer = torch.optim.Adam(self.network.parameters(),lr=lr)

        # Initialization
        X,Y=dataset[:][:]
        # Scale the data
        if scale:
            input_scaler = StandardScaler()
            X = input_scaler.fit_transform(X)
            output_scaler = StandardScaler()
            Y = output_scaler.fit_transform(Y)
            self.scalers = [input_scaler,output_scaler] # The model will now automatically scale the input data
        # Set up the bandwidth of the MMD
        Loss = MMD_loss(kernel)
        Loss.median_heuristic(Y,subsamples=1e4)
        loss_fn = Loss.loss_fn
        # Compute the initial loss
        idx = get_subsample_index(X,subsamples=1000)
        X, Y = X[idx], Y[idx]
        loss = loss_fn(self,X,Y)
        list_epoch = [0]
        list_loss = [loss.detach().numpy()]
        best_network = copy.deepcopy(self.network)
        best_loss_value = list_loss[-1]
        print(f"Epoch 0/{epochs}, Loss: {loss.detach().numpy()}")

        #Traing loop
        for epoch in range(1,epochs+1):
            for i, (x,y) in enumerate(dataloader):
                if scale:
                    y = output_scaler.transform(y) # The output data is scaled for training purposes
                #Generator update
                loss = loss_fn(self,x,y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_value = loss.detach().numpy()

            # At the end of one epoch
            if loss_value < best_loss_value:
                best_network = copy.deepcopy(self.network)
                best_loss_value = loss_value
            list_loss.append(loss_value)
            list_epoch.append(epoch)
            if epoch % 20 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss.detach().numpy()}")

        # At the end of all epoches
        if show:
            plt.plot(list_epoch,list_loss)
            plt.plot(list_epoch,np.zeros_like(list_epoch),color='red')
            plt.xlabel("Training steps")
            plt.ylabel("Training loss")
            plt.show()

        print(f"Best loss value: {best_loss_value}")
        self.network = best_network