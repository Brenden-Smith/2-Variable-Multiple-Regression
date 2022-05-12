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

# Import scipy to calculate the p-value for hypothesis testing
from scipy import stats

# Import Formulas class from formulas.py
from formulas import Formulas

def test_hypothesis(beta, name, b, se, n, alpha):
  '''
  Test the hypothesis of linearity b = 0 vs b != 0 at significance level alpha
  
  Parameters:
    beta (int): The beta value
    name (str): The name of the beta value
    b (int): The hypothesized value of beta
    se (int): The standard error of the beta value
    n (int): The number of data points
    alpha (int): the significance level
  '''
  
  # Calculate the t-statistic 
  t_stat = (beta - b) / se
  
  print("t-statistic:", t_stat, "with", n-2, "degrees of freedom")
  
  # Calculate the p-value
  p_value = stats.t.sf(np.abs(t_stat), n - 1) * 2
  
  # If the p-value is less than alpha, reject the null hypothesis. Else, accept the null hypothesis.
  if p_value < alpha: print("Reject the hypothesis", name, "=", b, "at", alpha, "with a p-value of", p_value) 
  else: print("Accept the hypothesis", name, "=", b, "at", alpha, "with a p-value of", p_value) 

def main():
  '''
  Main function
  '''
  
  path = "data_125.csv"
  # path = "data_150.csv"
  
  # Import data from csv file using pandas
  data = pd.read_csv(path)
  print(path, "imported successfully")
  
  ### Data preprocessing ###
  
  # Plot data points on a 3D graph using matplotlib
  fig = plt.figure()
  ax= fig.add_subplot(111, projection='3d')
  ax.scatter(data["x1"], data["x2"], data["y"])
  
  # Label the axes
  ax.set_xlabel("Critic Score")
  ax.set_ylabel("User Score")
  ax.set_zlabel("Album Popularity")

  # Create X and Y matrices
  X_mat = np.column_stack(([1] * len(data["x1"]), data["x1"], data["x2"]))
  Y_mat = np.reshape(np.array(data["y"]), (len(data["y"]), 1))
  
  ### Calculations ###
  
  # Normal: Calculate beta0, beta1, and beta2
  beta0, beta1, beta2 = Formulas.beta(X_mat, Y_mat)
  print("beta0:", beta0)
  print("beta1:", beta1)
  print("beta2:", beta2)
  print()
  
  # Calculate the regression values for y
  Y_reg = []
  for i in range(len(data["x1"])):
    Y_reg.append(Formulas.regression_line(data["x1"][i], data["x2"][i], beta0, beta1, beta2))
  
  # Calculate SSE vector
  sse = Formulas.sse(data["y"], Y_reg)
  print("SSE:", sse)
  print()
  
  # Calculate MSE
  mse = Formulas.mse(len(data["y"]), sse)
  print("MSE:", mse)
  print()
  
  # Calculate the standard errors of beta0, beta1, and beta2
  se_beta0, se_beta1, se_beta2 = Formulas.se_beta(mse, X_mat)
  print("se_beta0:", se_beta0)
  print("se_beta1:", se_beta1)
  print("se_beta2:", se_beta2)
  print()
  
  # Calculate the coefficient of correlation between x1 and y and x2 and y
  x1_corrcoeff = Formulas.coefficient_of_correlation(data["x1"], data["y"])
  x2_corrcoeff = Formulas.coefficient_of_correlation(data["x2"], data["y"])
  print("x1_corrcoeff:", x1_corrcoeff)
  print("x2_corrcoeff:", x2_corrcoeff)
  print()
  
  ### Hypothesis testing for linearity of beta1 and beta2 at 0.01 significance level ###
  
  # Test beta1 = 0 vs beta1 != 0
  test_hypothesis(beta1, "beta1", 0, se_beta1, len(data["y"]), 0.01)
  print()
  
  # Test beta2 = 0 vs beta2 != 0
  test_hypothesis(beta2, "beta2", 0, se_beta2, len(data["y"]), 0.01)
  print()
  
  # Plot the plane of best fit
  x1 = np.linspace(min(data["x1"]), max(data["x1"]), 100)
  x2 = np.linspace(min(data["x2"]), max(data["x2"]), 100)
  x1, x2 = np.meshgrid(x1, x2)
  ax.plot_surface(x1, x2, Formulas.regression_line(x1, x2, beta0, beta1, beta2), color='r', alpha=0.5)
  
  # Save graph image
  ax.view_init(elev=12, azim=45)
  fig.savefig('./img/%s.png' % path)
    
  # Show the graph
  plt.show()

# Execute main function
if __name__ == "__main__":
  main()