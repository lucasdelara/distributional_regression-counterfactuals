# Distributional regression for counterfactual reasoning

This is the code used for the simulations of the paper *Canonical Representations of Markovian Structural Causal Models: A Framework for Counterfactual Reasoning* (Lucas De Lara, 2025).

By executing the three scripts, one recovers the plots from the paper.

The code permits to train a distributional regression model in a similar fashion to standard regression models.

```
DR_model = distributional_regression_model(n_inputs=n_inputs,n_layers=4,hidden_size=64)

DR_model.optimize(dataset=dataset,epochs=40,batch_size=64,lr=1e-4,scale=True)
```

Such a model can be used, in particular, to compute counterfactual joint probability distributions. Interestingly, once the model is learned, one can test many types of counterfactual conceptions without extra inference steps, just by making a choice. Below, one simply switches between "Comonotonic" and "Stochastic".

```
plot_counterfactual_couplings(x_test,y_test,x_l,x_r,n_samples,DR_model,"Comonotonic",path)

plot_counterfactual_couplings(x_test,y_test,x_l,x_r,n_samples,DR_model,"Stochastic",path)
```

Several parts of the code are copied, adapted, or inspired by https://github.com/HWDance/Cocycles and https://github.com/airtai/monotonic-nn.
