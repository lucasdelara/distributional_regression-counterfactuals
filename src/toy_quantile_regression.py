import torch

import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import laplace

from utils.distributional_regression import *
from utils.data import *
from utils.basics import *

# Reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Generate test data
dataset = ToyDataSet(n_samples=5000)
x_test, y_test = dataset[:,:]

# Setup
n_datasets = 10
quantiles = [0.05,0.5,0.95]
metrics_gen = np.zeros((n_datasets,len(quantiles)))
metrics_reg = np.zeros((n_datasets,len(quantiles)))
metrics_true = np.zeros((n_datasets,len(quantiles)))
common_params = dict(learning_rate=0.05,n_estimators=200,max_depth=2,min_samples_leaf=5,min_samples_split=20,)
for i in range(n_datasets):
    # Generate a new dataset
    dataset = ToyDataSet(n_samples=5000)
    x_train, y_train = dataset[:,:]
    # Train the distributional regression model. It is the same for the three quantiles.
    DR_model = distributional_regression_model(n_inputs=1,n_layers=4,hidden_size=64, input_noise="Uniform")
    DR_model.optimize(dataset=dataset, epochs=40, batch_size=64, lr=1e-4,scale=True)
    gbr_models = []
    for j in range(len(quantiles)):
        # Train the regression model
        gbr = GradientBoostingRegressor(loss="quantile", alpha=quantiles[j], **common_params)
        gbr.fit(x_train, y_train.ravel())
        # Test the models
        e = quantiles[j]*torch.ones_like(x_test)
        y_gen = DR_model.det_forward(e,x_test).detach()
        y_reg = gbr.predict(x_test)
        y_true = mean_signal(x_test[:,0])+laplace.ppf(quantiles[j])
        metrics_gen[i,j] = mean_pinball_loss(y_test, y_gen, alpha=quantiles[j])
        metrics_reg[i,j] = mean_pinball_loss(y_test, y_reg, alpha=quantiles[j])
        metrics_true[i,j] = mean_pinball_loss(y_test, y_true, alpha=quantiles[j])
        gbr_models.append(gbr)

# Show metrics
means_gen = np.mean(metrics_gen,axis=0)
means_reg = np.mean(metrics_reg,axis=0) 
means_true = np.mean(metrics_true,axis=0)
std_gen = np.std(metrics_gen,axis=0)
std_reg = np.std(metrics_reg,axis=0)
std_true = np.std(metrics_true,axis=0)

df = pd.DataFrame(columns=quantiles, index=['gen','reg','true'])
df.loc['gen'] = means_gen
df.loc['reg'] = means_reg
df.loc['true'] = means_true
print(df)
df.to_csv('outputs/graphics/toy_quantile_means.csv')

df = pd.DataFrame(columns=quantiles, index=['gen','reg','true'])
df.loc['gen'] = 0.5*std_gen
df.loc['reg'] = 0.5*std_reg
df.loc['true'] = 0.5*std_true
print(df)
df.to_csv('outputs/graphics/toy_quantile_stds.csv') 

# Visualize
path = "outputs/graphics/toy_quantiles_curves.png"
n_steps = 50
x_grid = torch.linspace(start=0, end=10, steps=n_steps).reshape(-1,1)
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(9, 4))
for i in range(len(quantiles)):
    e = quantiles[i]*torch.ones_like(x_grid)
    y_gen = DR_model.det_forward(e,x_grid,rescale="True").detach()
    axs[0].scatter(x_grid[:,0], y_gen[:,0], alpha=1, s=0.5, marker='*', label="$q=${}".format(quantiles[i]))
    axs[0].set_title("Distributional regression")
    y_reg = gbr_models[i].predict(x_grid)
    axs[1].scatter(x_grid[:,0], y_reg, alpha=1, s=0.5, marker='*', label="$q=${}".format(quantiles[i]))
    axs[1].set_title("Gradient Boosting")
    y_true = mean_signal(x_grid[:,0])+laplace.ppf(quantiles[i])
    axs[2].scatter(x_grid[:,0], y_true, alpha=1, s=0.5, marker='*', label="$q=${}".format(quantiles[i]))
    axs[2].set_title("True curves")
for ax in axs:
    ax.scatter(x_test, y_test, alpha=0.005, color='grey')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
leg = axs[2].legend()
for lh in leg.legend_handles:
    lh.set_alpha(1)
    lh._sizes = [8]
fig.tight_layout()
plt.savefig(path)
plt.show()
