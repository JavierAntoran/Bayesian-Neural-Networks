# Bayesian Neural Networks

Pytorch implementations for the following approximate inference methods:

* [Bayes by Backprop](#bayes-by-backprop)
* [Bayes by Backprop + Local Reparametrisation Trick](#local-reparametrisation-trick)
* [MC dropout](#mc-dropout)
* [Stochastic Gradient Langevin Dynamics](#stochastic-gradient-langevin-dynamics)
* [Preconditioned SGLD](#pSGLD)
* [Bootstrap MAP Ensemble](#bootstrap-map-ensemble)

### Prerequisites
* PyTorch (https://pytorch.org)
* Numpy
* Matplotlib

## Approximate inference in Neural Networks

## Bayes by Backprop (BBP)
(https://arxiv.org/abs/1505.05424)

Train a model on MNIST:
```bash
python train_BayesByBackprop_MNIST.py [--model [MODEL]] [--prior_sig [PRIOR_SIG]] [--epochs [EPOCHS]] [--lr [LR]] [--n_samples [N_SAMPLES]] [--models_dir [MODELS_DIR]] [--results_dir [RESULTS_DIR]]
```
For an explanation of the script's arguments:
```bash
python train_BayesByBackprop_MNIST.py -h
```



Best results are obtained with a Laplace prior.

### Local Reparametrisation Trick
(https://arxiv.org/abs/1506.02557)

Bayes By Backprop inference where the mean and variance of activations
 are calculated in closed form. Activations are sampled instead of
 weights. This makes the variance of the Monte Carlo ELBO estimator scale
 as 1/M, where M is the minibatch size. Sampling weights scales (M-1)/M.
 The KL divergence between gaussians can also be computed in closed form,
 further reducing variance. Computation of each epoch is faster and so is convergence.

Train a model on MNIST:
```bash
python train_BayesByBackprop_MNIST.py --model Local_Reparam [--prior_sig [PRIOR_SIG]] [--epochs [EPOCHS]] [--lr [LR]] [--n_samples [N_SAMPLES]] [--models_dir [MODELS_DIR]] [--results_dir [RESULTS_DIR]]
```

## MC Dropout
(https://arxiv.org/abs/1506.02142)

A fixed dropout rate of 0.5 is set.

Train a model on MNIST:
```bash
python train_MCDropout_MNIST.py [--weight_decay [WEIGHT_DECAY]] [--epochs [EPOCHS]] [--lr [LR]] [--models_dir [MODELS_DIR]] [--results_dir [RESULTS_DIR]]
```
For an explanation of the script's arguments:
```bash
python train_MCDropout_MNIST.py -h
```


## Stochastic Gradient Langevin Dynamics (SGLD)
(https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf)

In order to converge to the true posterior over w, the learning rate
should be annealed according to the [Robbins-Monro](https://en.wikipedia.org/wiki/Stochastic_approximation)
 conditions. In practise, we use a fixed learning rate.

Train a model on MNIST:
```bash
train_SGLD_MNIST.py [--use_preconditioning [USE_PRECONDITIONING]] [--prior_sig [PRIOR_SIG]] [--epochs [EPOCHS]] [--lr [LR]] [--models_dir [MODELS_DIR]] [--results_dir [RESULTS_DIR]]
```
For an explanation of the script's arguments:
```bash
python train_SGLD_MNIST.py -h
```

### pSGLD
(https://arxiv.org/abs/1512.07666)

SGLD with RMSprop preconditioning. A higher learning rate should be used
than for vanilla SGLD.

Train a model on MNIST:
```bash
train_SGLD_MNIST.py --use_preconditioning True [--prior_sig [PRIOR_SIG]] [--epochs [EPOCHS]] [--lr [LR]] [--models_dir [MODELS_DIR]] [--results_dir [RESULTS_DIR]]
```

## Bootstrap MAP Ensemble

Multiple networks are trained on subsamples of the dataset.

## Results

### MNIST Classification

W is marginalised with 100 samples of the weights for all models except
MAP, where only one set of weights is used.

|      MNIST     	|   MAP   	| MAP  Ensemble 	| BBBP  Gaussian 	| BBBP  GMM 	| BBBP  Laplace 	| BBBP Local Reparam 	| MC Dropout 	|   SGLD  	|  pSGLD  	|
|:--------------:	|:-------:	|:-------------------:	|:--------------:	|:---------:	|:-------------:	|:-------------------:	|:----------:	|:-------:	|:-------:	|
| Log Like	| -572.90 	|       -496.54       	|    -1100.29    	|  -1008.28 	|    -892.85    	|       -1086.43      	|  -435.458  	| -828.29 	| -661.25 	|
|    Error \%    	|   1.58  	|         1.53        	|      2.60      	|    2.38   	|      2.28     	|         2.61        	|    1.37    	|   1.76  	|   1.76  	|

### MNIST Uncertainty

### Homoscedastic Regression

### Heteroscedastic Regression

### Weight Distributions

### Weight Pruning