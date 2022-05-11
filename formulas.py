# Brenden Smith & Jason Huynh
# EE 381 Regression Project
# 12 May 2022

# Numpy is imported to handle various math functions
import numpy as np

class Formulas:
  '''
  This is a class that handles the important formulas to calculate multiple regression.
  There are no instance variables, all methods are static.
  '''
  
  # Regression line
  def regression_line(x1, x2, beta0, beta1, beta2):
    '''
    Regression line formula
    
    Calculate y^ values given x1, x2, beta0, beta1, and beta2
    
    Parameters:
      x1 (array | int): x1 value(s)
      x2 (array | int): x2 value(s)
      beta0 (int): beta0 value
      beta1 (int): beta1 value
      beta2 (int): beta2 value
      
    Returns:
      array | int: y^ value(s)
    '''
    return (beta1 * x1) + (beta2 * x2) + beta0
  
  def beta(x, y):
    '''
    Calculate beta0, beta1, and beta2 given x and y matrices
    
    Parameters:
      x (matrix): X matrix
      y (matrix): Y matrix
      
    Returns:
      array: beta0, beta1, and beta2
    '''
    
    # Important constants
    x_t = np.transpose(x)
    
    # Calculate beta0, beta1, and beta2
    result = np.matmul(np.linalg.pinv(np.matmul(x_t, x)), np.matmul(x_t, y))
    
    # Return beta0, beta1, and beta2
    return result[0][0], result[1][0], result[2][0]

  def sse(y, y_reg):
    '''
    Calculate the sum of squares error given y and y_reg values
    
    Parameters:
      y (array): y values
      y_reg (array): y^ values
    
    Returns:
      int: Sum of squares error
    '''
    
    # Calculate matrix composed of  (y - y_reg)
    b = []
    for i in range(len(y)):
      b.append(y[i] - y_reg[i])
    
    # Create (y - y_reg) transposed matrix
    b = np.array(b)
    a = np.transpose(np.array(b))
    
    # Return sum of squares error
    return np.matmul(a, b)
    
  def mse(n, sse):
    '''
    Calculate the mean square error given n and sse values
    
    Parameters:
      n (int): Size of matrix
      sse (int): Sum of squares error
      
    Returns:
      int: Mean square error
    '''
    return sse / (n - 3)

  def se_beta(mse, x):
    '''
    Calculate the standard error for beta0, beta1, and beta2
    
    Parameters:
      mse (int): Mean square error
      x (array): x values
    
    Returns:
      array: Standard error for beta0, beta1, and beta2
    '''
    
    # Initialize array
    beta = []
    
    # Calculate matrix composed of X'X
    mat = np.linalg.pinv(np.matmul(np.transpose(x), x))
    
    # Calculate standard error for beta0, beta1, and beta2
    for i in range(len(mat)):
      beta[i] = np.sqrt(mse * mat[i][i])
      
    # Return standard error for beta0, beta1, and beta2
    return beta
  
  def coefficient_of_correlation(x, y):
    '''
    Calculate the coefficient of correlation given x and y arrays
    
    Parameters:
      x (array): x values
      y (array): y values
    
    Returns:
      int: Coefficient of correlation
    '''
    
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
    
