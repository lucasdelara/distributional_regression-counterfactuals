import torch
from torch.distributions.multivariate_normal import MultivariateNormal

import numpy as np

import matplotlib.pyplot as plt

from utils.distributional_regression import *
from utils.data import *
from utils.basics import *

# Reproducibility
torch.manual_seed(0)
np.random.seed(0)
  
# Global variables
show_data = False # Whether to plot the training set
train = True # Whether to train a new DR model or to use a saved one

# Generate and visualize the training data
dataset = ToyDataSet(n_samples=5000)
x_train, y_train = dataset[:,:]
n_inputs = 1

# Visualisation
if show_data:
    x_grid = np.linspace(0,10,100)
    mean_y = mean_signal(x_grid)
    plt.scatter(x_train[:,0], y_train[:,0], alpha=0.5, label="Train samples")
    plt.plot(x_grid, mean_y, label="Mean curve", color='brown')
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()
    plt.show()

# Distributional regression on training data
path = "outputs/models/toy.pt"
if train:
    DR_model = distributional_regression_model(n_inputs=n_inputs,n_layers=4,hidden_size=64)
    DR_model.optimize(dataset=dataset, epochs=40, batch_size=64, lr=1e-4,scale=True)
    torch.save(DR_model, path)
else:
    DR_model = torch.load(path, weights_only=False)
DR_model.network.eval()

# Generate test data
dataset = ToyDataSet(n_samples=5000)
x_test, y_test = dataset[:,:]

# Evaluation of the DR model for statistics

# Ev 1: Ability to generate the joint distribution
generated_y = DR_model.sto_forward(x_test, rescale=True).detach()

path = "outputs/graphics/toy_joint"
plt.scatter(x_test[:,0], y_test[:,0], alpha=0.5, label="True samples")
plt.scatter(x_test[:,0], generated_y[:,0], alpha=0.2, label="Generated samples")
plt.xlabel("$x$")
plt.ylabel("$y$")
leg = plt.legend()
for lh in leg.legend_handles:
    lh.set_alpha(1)
plt.title("$P_{\mathtt{x,y}}$")
plt.savefig(path)
plt.show()

# Ex 2: Ability to generate the conditional distribution for x=5
x = torch.tensor([5.0])
generated_y = DR_model.sample(x,1000, rescale=True).detach()

# From above
path = "outputs/graphics/toy_conditional"
plt.scatter(x_test, y_test, alpha=0.5, label="True samples")
plt.scatter(x*torch.ones((1000,1)), generated_y[0], alpha=0.5, label="Generated samples at $x=5$")
plt.xlabel("$x$")
plt.ylabel("$y$")
leg = plt.legend()
for lh in leg.legend_handles:
    lh.set_alpha(1)
plt.savefig(path)
plt.show()

# Histogram
mean = mean_signal(5.0)
bins = 20
out = torch.histogram(generated_y, bins=bins, density=True)
hist = out.hist
bin_edges = out.bin_edges
bin_coord = hist_coord(bin_edges)
x2_axis = torch.linspace(-10 + mean,10 + mean,100)
true_curve = (1/(np.sqrt(2*np.pi)))*np.exp(-(x2_axis-mean)**2/2)

path = "outputs/graphics/toy_hist"
plt.bar(bin_coord, hist, label="Estimated density", color="orange", align='center')
plt.plot(x2_axis, true_curve, alpha=0.8, label="True density", color='blue')
plt.xlabel("$y$")
plt.ylabel("$Density$")
leg = plt.legend()
for lh in leg.legend_handles:
    lh.set_alpha(1)
plt.title("$P_{\mathtt{y}|\mathtt{x}}(\cdot|x)$ at $x=5$")
plt.savefig(path)
plt.show()

# Ev 3: Plotting as a function of x for a single e
e0 = torch.zeros_like(x_test)
generated_y0 = DR_model.det_forward(e0,x_test,rescale=True).detach()
em = -0.5*torch.ones_like(x_test)
generated_ym = DR_model.det_forward(em,x_test,rescale=True).detach()
ep = torch.ones_like(x_test)
generated_yp = DR_model.det_forward(ep,x_test,rescale=True).detach()

path = "outputs/graphics/toy_e_curves"
plt.scatter(x_test, y_test, alpha=0.5, label="True samples")
plt.scatter(x_test[:,0], generated_yp[:,0], alpha=0.5, s=0.5, color="purple", label="Generated samples for $e=1$")
plt.scatter(x_test[:,0], generated_y0[:,0], alpha=0.5, s=0.5, color="red", label="Generated samples for $e=0$")
plt.scatter(x_test[:,0], generated_ym[:,0], alpha=0.5, s=0.5, color="pink", label="Generated samples for $e=-0.5$")
plt.xlabel("$x$")
plt.ylabel("$y$")
leg = plt.legend()
for lh in leg.legend_handles:
    lh.set_alpha(1)
    lh._sizes = [8]
plt.title("Deterministic curves")
plt.savefig(path)
plt.show()

## Ev 4: Plotting as a function of e for x=5
e_grid = torch.linspace(start=-5, end=5, steps=100).reshape(-1,1)
x_val = 5.0*torch.ones_like(e_grid)

generated_y = DR_model.det_forward(e_grid,x_val,rescale=True).detach()
expected_y = mean_signal(5.0) + normal_to_laplace(e_grid)

path = "outputs/graphics/toy_psi"
plt.plot(e_grid,generated_y, label="$\hat{\psi}^{\mathcal{C}}_{\mathtt{y}}(\cdot|x)$")
plt.plot(e_grid,expected_y, label="$\psi^{\mathcal{C}}_{\mathtt{y}}(\cdot|x)$")
plt.xlabel("Input noise $e$")
plt.ylabel("Output $y$")
leg = plt.legend()
for lh in leg.legend_handles:
    lh.set_alpha(1)
plt.title('Increasing transport at $x=5$')
plt.savefig(path)
plt.show()

# Evaluation of the DR model for general causation

# Intervention
def phi(x):
    return torch.min(x+1,10*torch.ones_like(x))

# Causal effect
def phi_effect(x,n_samples=2000):
    y = DR_model.sample(x,n_samples,rescale=True).detach()
    y_phi = DR_model.sample(phi(x),n_samples,rescale=True).detach()
    return (y_phi - y).mean(dim=1), (y_phi - y).std(dim=1)

path = "outputs/graphics/toy_causal_effect"
x_grid = torch.linspace(start=0, end=10, steps=100).reshape(-1,1)
true_effect = mean_signal(phi(x_grid))-mean_signal(x_grid)
mean_effect, std_effect = phi_effect(x_grid)
x_grid = x_grid.ravel()
plt.plot(x_grid,true_effect,label="True function",color='blue')
plt.plot(x_grid,mean_effect,label='Estimated function',color='orange')
plt.fill_between(x_grid,mean_effect+0.5*std_effect,mean_effect-0.5*std_effect,color='orange',alpha=0.2)
plt.xlabel("x")
plt.ylabel("CATE($x$)")
plt.legend()
plt.savefig(path)
plt.show()

# Evaluation of the DR model for singular causation

# Functions to compute and plot counterfactual quantities

def individual_curve(x_grid,concept,model):
    n_steps = x_grid.size()[0]
    if concept=="Comonotonic":
        # Normalization
        e = torch.normal(mean=0, std=1, size=(1,))
        E = e*torch.ones_like(x_grid)
        # Transport of the normalization
        Y_curve = model.det_forward(E,x_grid,rescale="True").detach()
    elif concept=="Stochastic":
        sigma = 0.8
        K = gaussian_kernel(lengthscale=sigma)
        eps = 1e-5 # To force the covariance matrix to be positive definite
        x_grid = torch.linspace(start=0, end=10, steps=n_steps).reshape(-1,1)
        m_grid = torch.zeros(n_steps)
        K_grid = K.get_gram(x_grid,x_grid) + eps * torch.eye(n_steps)
        # Normalization
        normalized_process = MultivariateNormal(loc=m_grid,covariance_matrix=K_grid)
        E_indiv = normalized_process.sample().reshape(-1,1)
        # Transport of the normalization
        Y_curve = model.det_forward(E_indiv,x_grid,rescale="True").detach()
    else:
        raise Exception("This concept is not implemented. Choose between 'Comonotonic' and 'Stochastic'.")
    return Y_curve

def counterfactual_coupling(x_l,x_r,n_samples,model,concept):
    if concept=="Comonotonic":
        E = torch.normal(mean=0,std=1,size=(n_samples,1))
        x_r = x_r*torch.ones((n_samples,1))
        Y_r = model.det_forward(E,x_r,rescale='True').detach()
        x_l = x_l*torch.ones((n_samples,1))
        Y_l = model.det_forward(E,x_l,rescale='True').detach()
    elif concept=='Stochastic':
        sigma = 0.8
        K = gaussian_kernel(lengthscale=sigma)
        x_lr = torch.tensor([x_l,x_r]).reshape(-1,1)
        m_lr = torch.zeros(2)
        K_lr = K.get_gram(x_lr,x_lr)
        normalized_coupling = MultivariateNormal(loc=m_lr,covariance_matrix=K_lr)
        E_lr = normalized_coupling.sample((n_samples,))
        E_l, E_r = E_lr[:,0].reshape(-1,1), E_lr[:,1].reshape(-1,1)
        x_l = x_l*torch.ones_like(E_l)
        x_r = x_r*torch.ones_like(E_r)
        Y_l = model.det_forward(E_l,x_l,rescale='True').detach()
        Y_r = model.det_forward(E_r,x_r,rescale='True').detach()
    else:
        raise Exception("This concept is not implemented. Choose between 'Comonotonic' and 'Stochastic'.")
    return Y_l,Y_r

def plot_individual_curves(x_data,y_data,x_grid,concept,model,n_indiv,path):
    # Data
    plt.scatter(x_data, y_data, alpha=0.02, color='blue', label="Population",s=10)
    # Individual
    for i in range(n_indiv):
        y_curve = individual_curve(x_grid,concept,model)
        if i==0:
            plt.plot(x_grid, y_curve, linestyle='solid', marker='o', alpha=1,linewidth=0.5, markersize=0.5, color='red',label='Individual')
        else:
            plt.plot(x_grid, y_curve, linestyle='solid', marker='o', alpha=1,linewidth=0.5, markersize=0.5)
    # Plot
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0,10)
    plt.ylim(-8,18)
    leg = plt.legend()
    for lh in leg.legend_handles:
        lh.set_alpha(1)
        lh._sizes = [8]
    plt.savefig(path)
    plt.show()

def plot_counterfactual_couplings(x_data,y_data,x_l,x_r,n_samples,model,concept,path):
    Y_l,Y_r = counterfactual_coupling(x_l,x_r,n_samples,model,concept)
    ## Observational
    plt.scatter(x_data[:,0], y_data[:,0], alpha=0.05, color='blue', label="True data", s=10)
    ## Interventional
    plt.scatter(np.repeat(x_l,20,axis=0), Y_l[:], alpha=0.6, color = 'purple', label="Obs/Int",s=10)
    plt.scatter(np.repeat(x_r,20,axis=0), Y_r[:], alpha=0.6, color = 'purple',s=10)
    ## Counterfactual
    plt.arrow(x_l, Y_l[0,0], x_r-x_l, Y_r[0,0]-Y_l[0,0], alpha=0.4, color='red', label="Ctf")
    for i in range(1,n_samples):
        plt.arrow(4, Y_l[i,0], x_r-x_l, Y_r[i,0]-Y_l[i,0], alpha=0.4, color='red')
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.xlim(0,10)
    plt.ylim(-8,18)
    leg = plt.legend()
    for lh in leg.legend_handles:
        lh.set_alpha(1)
        lh._sizes = [8]
    plt.savefig(path)
    plt.show()

# Process parameters
n_steps = 50 # Number of parallel worlds
x_grid = torch.linspace(start=0, end=10, steps=n_steps).reshape(-1,1)
n_indiv = 10 # Number of individuals

# Coupling parameters
x_l = 4.0 # Left marginal at x=4
x_r = 6.0 # Right marginal at x=6
n_samples = 20

# Trying two counterfactual conceptions. This does not require any inferential step: it only demands to specify a concept.

# Comonotonic counterfactuals

# Individual curves
path = "outputs/graphics/toy_comonotonic_indiv"
plot_individual_curves(x_test,y_test,x_grid,"Comonotonic",DR_model,n_indiv,path)
# Coupling
path = "outputs/graphics/toy_comonotonic_coupling"
plot_counterfactual_couplings(x_test,y_test,x_l,x_r,n_samples,DR_model,"Comonotonic",path)

# Stochastic comonotonic counterfactuals

# Individual curves
path = "outputs/graphics/toy_gaussian_indiv"
plot_individual_curves(x_test,y_test,x_grid,"Stochastic",DR_model,n_indiv,path)
# Coupling
path = "outputs/graphics/toy_gaussian_coupling"
plot_counterfactual_couplings(x_test,y_test,x_l,x_r,n_samples,DR_model,"Stochastic",path)
