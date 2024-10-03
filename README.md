# ENOT: Elastic Net Optimal Transport
This is the official code repository of ACCV'24 oral paper "Sparse Domain Transfer via Elastic Net Regularization"

## Code Description
This repository contains the implementation of core algorithms of ENOT, and original codes to replicate the experimental results reported in the paper.

**We are refactoring the codebase to make it more user-friendly. To have a preview first, here is a general guideline about the code files**

1. *image_potential_func.py*: Containing the codes about training the neural network potential function in the image domain.
2. *imdb.py*: Containing the codes to produce our IMDB experimental results
3. *mnist.py*: Containing the codes to produce our MNIST experimental results
4. *syn.py*: Containing the codes to produce our synthetic Gaussian mixtures experimental results