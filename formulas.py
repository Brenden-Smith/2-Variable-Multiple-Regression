# Brenden Smith & Jason Huynh
# EE 381 Regression Project
# 12 May 2022

# MODULE IMPORTS
# Matplotlib for plotting the data onto a graph
import matplotlib.pyplot as plt

# Numpy for various math functions
import numpy as np

# Pandas for importing the data
import pandas as pd

# Important formulas
class Formulas:
  
  # Calculate beta0, beta1, and beta2 give x and y matrices
  def beta(x, y):
    x_t = np.transpose(x)
    return np.matmul(np.linalg.pinv(np.matmul(x_t, x)), np.matmul(x_t, y))

  # Calculate sum of squares error given x and y matrices and the regression line
  def sse(x, y, beta0, beta1, beta2):
    result = []
    for i in range(len(y)):
      result.append(y[i][0] - (beta0 + beta1 * x[i][0] + beta2 * x[i][0] ** 2))
    return result
    
  # Calculate error variance of the matrix given size and sum of squares error
  def mse(n, sse):
    return sse / (n - 3)

  # Calculate standard error for beta0, beta1, and beta2
  def se_beta(mse, x):
    result = []
    mat = np.linalg.inv(np.matmul(np.transpose(x), x))
    for i in range(len(mat)):
      result.append(np.sqrt(mse * mat[i][i]))
    return result
  
  # Calculate the coefficient of correlation given x and y arrays
  def coefficient_of_correlation(x, y):
    
    # Important constants
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate numerator
    numerator = 0
    for i in range(len(x)):
      numerator += x[i][0] - x_mean * (y[i][0] - y_mean)

    # Calculate denominator
    d1, d2 = 0
    for i in range(len(x)):
      d1  += (x[i][0] - x_mean) ** 2
      d2 += (y[i][0] - y_mean) ** 2
    denominator = np.sqrt(d1 * d2)
    
    # Return result
    return numerator / denominator
    
