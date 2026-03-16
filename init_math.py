import numpy as np
import matplotlib.pyplot as plt

# 1. Define our Loss Function (The Mountain: L = w^2)
def loss_function(w):
    return w ** 2

# 2. Define our Gradient (The Compass: derivative of w^2 is 2w)
def gradient(w):
    return 2 * w

# 3. The Gradient Descent Algorithm
def gradient_descent(start_w, learning_rate, epochs):
    w = start_w
    weight_history = []
    loss_history = []
    
    for i in range(epochs):
        # Save current state for our plot
        weight_history.append(w)
        loss_history.append(loss_function(w))
        
        # Calculate the gradient (compass reading)
        grad = gradient(w)
        
        # Take a step down the mountain
        w = w - (learning_rate * grad)
        
    return weight_history, loss_history

# --- Let's run it! ---
# Start high up on the mountain at w = 10
# Take moderate steps (learning rate = 0.1)
# Do this for 20 steps (epochs)
weights, losses = gradient_descent(start_w=10, learning_rate=0.1, epochs=20)

# --- Plotting the results ---
plt.plot(weights, losses, marker='o', color='red', label='Steps down the mountain')
plt.title("Gradient Descent Finding the Lowest Loss")
plt.xlabel("Weight (w)")
plt.ylabel("Loss (Error)")
plt.legend()
plt.grid()
plt.show()