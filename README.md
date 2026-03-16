# Gradient-Descent-from-Scratch
This is the mathimatical concept that I have learned for machine learning.
#  Gradient Descent from Scratch

This repository contains a from-scratch Python implementation of Gradient Descent, a foundational optimization algorithm used in Machine Learning. 

Currently, this project demonstrates how a model iteratively updates its weights to find the absolute minimum of a 1D loss function (`L = w^2`).

##  The Intuition

Instead of relying on black-box libraries, this implementation breaks down the calculus behind the algorithm:
* **The Mountain (Loss Function):** The error landscape we are trying to navigate down.
* **The Compass (Gradient/Derivative):** Calculates the steepness of the current position to find the direction of the steepest ascent. We multiply this by `-1` to go *down* the mountain.
* **The Stride (Learning Rate):** A hyperparameter that determines how big of a step to take. 

##  Running the Code

### Requirements
* `numpy`
* `matplotlib`

### Output
Running `gradient_descent.py` will calculate the steps down the loss curve and generate a plot showing the path the algorithm took to converge at an error of 0. Notice how the steps naturally get smaller as the slope flattens out!

##  Future Roadmap
* [ ] Add Stochastic Gradient Descent (SGD)
* [ ] Add Adam Optimizer
