import torch
from torch.utils.data import random_split, Subset
from torch.distributions.multivariate_normal import MultivariateNormal

import numpy as np

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor

from utils.distributional_regression import *
from utils.data import *
from utils.basics import *

# Reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Function to generate counterfactual couplings
def counterfactual_coupling(model,Z,concept):
    n_samples = len(Z)
    T0 = torch.zeros((n_samples,1)) # Fixing T to 0
    T1 = torch.ones((n_samples,1)) # Fixing T to 1
    X0 = torch.hstack([T0,Z]) # Z is unchanged by interventions on T
    X1 = torch.hstack([T1,Z])
    if concept=='Comonotonic':
        E = torch.normal(mean=0,std=1,size=(n_samples,1))
        Y0 = model.det_forward(E,X0,rescale=True).detach()
        Y1 = model.det_forward(E,X1,rescale=True).detach()
    if concept=='Countermonotonic':
        E0 = torch.normal(mean=0,std=1,size=(n_samples,1))
        E1 = -E0
        Y0 = model.det_forward(E0,X0,rescale=True).detach()
        Y1 = model.det_forward(E1,X1,rescale=True).detach()
    if concept=='Stochastic':
        sigma = 2
        K = gaussian_kernel(lengthscale=sigma)
        t_01 = torch.tensor([0,1]).reshape(-1,1)
        m_01 = torch.zeros(2)
        K_01 = K.get_gram(t_01,t_01)
        normalized_coupling = MultivariateNormal(loc=m_01,covariance_matrix=K_01)
        E = normalized_coupling.sample((n_samples,))
        E0 = E[:,0].reshape(-1,1)
        E1 = E[:,1].reshape(-1,1)
        Y0 = model.det_forward(E0,X0,rescale=True).detach()
        Y1 = model.det_forward(E1,X1,rescale=True).detach()
    return Y0, Y1
  
# Global variables
train = True

# Load dataset
dataset = Fetch_401k_DataSet()
n_data, n_features = dataset.shape()
train_set, test_set = random_split(dataset, (7000,2915))
X_train, Y_train = train_set[:][0], train_set[:][1]
X_test, Y_test = test_set[:][0], test_set[:][1]
n_samples = 2915
n_inputs = X_train.size()[1]

# Training the normalizing map
path = "outputs/models/e401k.pt"
if train:
   DR_model = distributional_regression_model(n_inputs=n_inputs,n_layers=2,hidden_size=128)
   DR_model.optimize(dataset=train_set, scale=True, epochs=40, batch_size=128, lr=1e-3)
   torch.save(DR_model, path)
else:
    DR_model = torch.load(path, weights_only=False)
DR_model.network.eval()

# Visualizing three counterfactual conceptions

Z = X_test[:,1:] # Unfixed Z, to study (Y0,Y1)
z_mean = torch.mean(X_test[:,1:], dim=0).repeat((n_samples,1)) # Fixed Z, to study (Y0,Y1)|Z=z
concepts = ['Comonotonic','Countermonotonic','Stochastic']
colors = ['red','orange','purple']

# Counterfactual coupling (Y0,Y1)|Z=z
path = "outputs/graphics/e401_k_counterfactuals_at_z"
for i in range(len(concepts)):
    Y0, Y1 = counterfactual_coupling(DR_model,z_mean,concepts[i])
    plt.scatter(Y0[:,0], Y1[:,0], alpha=0.3, s=1, label=concepts[i], color=colors[i])
plt.xlabel("$Y_0$")
plt.ylabel("$Y_1$")
leg = plt.legend()
for lh in leg.legend_handles:
    lh.set_alpha(1)
    lh._sizes = [8]
plt.savefig(path)
plt.show()

# Counterfactual coupling (Y0,Y1)
path = "outputs/graphics/e401_k_counterfactuals"
outcomes = []
for i in range(len(concepts)):
    Y0, Y1 = counterfactual_coupling(DR_model,Z,concepts[i])
    outcomes.append([Y0,Y1])
    plt.scatter(Y0[:,0], Y1[:,0], alpha=0.3, s=1, label=concepts[i], color=colors[i])
plt.xlabel("$Y_0$")
plt.ylabel("$Y_1$")
leg = plt.legend()
for lh in leg.legend_handles:
    lh.set_alpha(1)
    lh._sizes = [8]
plt.savefig(path)
plt.show()

# Counterfactual effect estimation
quantiles = np.linspace(0.01,0.99,100)
path = "outputs/graphics/e401_k_effects"
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))
n_bootstrap = 20
effects = [[] for b in range(n_bootstrap)]
for b in range(n_bootstrap):
    # For each bootstrap replication, learn a DR model on the training set
    bootstrap_inds = torch.randint(0,n_data-1,(n_data,))
    bootstrap_dataset = Subset(dataset,bootstrap_inds)
    train_set, test_set = random_split(bootstrap_dataset, (7000,2915))
    X_train, Y_train = train_set[:][0], train_set[:][1]
    X_tes, Y_test = test_set[:][0], test_set[:][1]
    DR_model = distributional_regression_model(n_inputs=n_inputs,n_layers=2,hidden_size=128)
    DR_model.optimize(dataset=train_set, scale=True, epochs=40, batch_size=128, lr=1e-3)
    DR_model.network.eval()
    Zb = X_test[:,1:]
    for i in range(len(concepts)):
        # For each counterfactual coupling generated on the test set, learn a MSE regression model and compute the causal effect
        Y0,Y1 = counterfactual_coupling(DR_model,Zb,concepts[i])
        Y0,Y1 = Y0.numpy(),Y1.numpy()
        scale_0 = np.std(Y0.ravel())
        scale_1 = np.std(Y1.ravel())
        Y0 = Y0.reshape(-1,1)
        Y0s = Y0/scale_0
        Y1s = (Y1/scale_1).ravel()
        CE_model = MLPRegressor(max_iter=600)
        CE_model.fit(Y0s,Y1s)
        Y0_q = np.quantile(Y0.ravel(),quantiles)
        Y1_pred_q = CE_model.predict(Y0_q.reshape(-1,1)/scale_0)*scale_1
        effect = Y1_pred_q - Y0_q.ravel()
        effects[b].append(effect)

effects = np.array(effects)
mean_pred = np.mean(effects,axis=0)
std_pred = np.std(effects,axis=0)

for i in range(len(concepts)):
    axs[i].plot(quantiles, mean_pred[i], color=colors[i])
    axs[i].fill_between(quantiles, mean_pred[i]-0.5*std_pred[i], mean_pred[i]+0.5*std_pred[i], color=colors[i], alpha=0.2)
    axs[i].set_title(concepts[i])
    axs[i].set_xlabel("q")
axs[0].set_ylabel("$\mathbb{E}[Y_1 - Y_0 | Y_0=G_0(q)]$")
fig.tight_layout()
plt.savefig(path)
plt.show()