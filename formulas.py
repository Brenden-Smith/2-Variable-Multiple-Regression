# Brenden Smith & Jason Huynh
# EE 381 Regression Project
# 12 May 2022

# MODULE IMPORTS
# Numpy for various math functions
import numpy as np

# Important formulas
class Formulas:
  
  # Regression line
  def regression_line(x1, x2, beta0, beta1, beta2):
    return (beta1 * x1) + (beta2 * x2) + beta0
  
  # Calculate beta0, beta1, and beta2 give x and y matrices
  def beta(x, y):
    x_t = np.transpose(x)
    result = np.matmul(np.linalg.pinv(np.matmul(x_t, x)), np.matmul(x_t, y))
    return result[0][0], result[1][0], result[2][0]

  # Calculate sum of squares error given y and y_reg values
  def sse(y, y_reg):
    b = []
    for i in range(len(y)):
      b.append(y[i] - y_reg[i])
    b = np.array(b)
    a = np.transpose(np.array(b))
    return np.matmul(a, b)
    
  # Calculate error variance of the matrix given size and sum of squares error
  def mse(n, sse):
    return sse / (n - 3)

  # Calculate standard error for beta0, beta1, and beta2
  def se_beta(mse, x):
    result = []
    mat = np.linalg.pinv(np.matmul(np.transpose(x), x))
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
      numerator += x[i] - x_mean * (y[i] - y_mean)

    # Calculate denominator
    d1 = d2 = 0
    for i in range(len(x)):
      d1  += (x[i] - x_mean) ** 2
      d2 += (y[i] - y_mean) ** 2
    denominator = np.sqrt(d1 * d2)
    
    # Return result
    return numerator / denominator
    
