
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 5.0)

#Simple 2D gradient descent linear regression
#Higher dimension generalization could be implemented via vectorization

data = pd.read_csv('data.csv') #Random data from internet
X = data.iloc[:, 0] # X values
Y = data.iloc[:, 1] # Y values

theta0 = 0 
theta1 = 0  #hyperparameters

alpha = 0.0001  # Step size
iteration = 1000  # The number of iterations to perform gradient descent

n = float(len(X)) # Number of elements 

# Performing Gradient Descent 
for i in range(iteration): 
    Y_pred = theta0 + theta1*X  # The current predicted value of Y
    D_t1 = (-2/n) * sum(X * (Y - Y_pred))  # Derivative of cost func. with respect to thetas
    D_t0 = (-2/n) * sum(Y - Y_pred)  
    theta0 = theta0 - alpha * D_t0  # Update thetas
    theta1 = theta1 - alpha * D_t1  
    
 

Y_pred = theta1*X + theta0

plt.scatter(X, Y) 
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='black')  # regression line
plt.show()
