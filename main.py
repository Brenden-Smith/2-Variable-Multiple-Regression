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

# Import Formulas class from formulas.py
from formulas import Formulas

# Main function
def main():
  
  # Import data from csv file using pandas
  data = pd.read_csv("data.csv")
  
  # Plot data points on graph using matplotlib
  plt.plot(data["x1"], data["y"], "ro")
  plt.plot(data["x2"], data["y"], "ro")
  
  # Create X and Y matrices
  X_mat = np.column_stack(([1] * len(data["x1"]), data["x1"], data["x2"]))
  Y_mat = np.reshape(np.array(data["y"]), (len(data["y"]), 1))
  
  # Calculate beta0, beta1, and beta2
  beta0, beta1, beta2 = Formulas.beta(X_mat, Y_mat)
  print(beta0, beta1, beta2)
  
  # Calculate SSE vector
  sse = Formulas.sse(X_mat, Y_mat, beta0, beta1, beta2)
  print(sse)
  
  # Calculate MSE
  mse = Formulas.mse(len(data["y"]), sse)
  print(mse)
  
  # Calculate the standard errors of beta0, beta1, and beta2
  se_beta0, se_beta1, se_beta2 = Formulas.beta(mse, X_mat)
  print(se_beta0, se_beta1, se_beta2)
  
  # Calculate the coefficient of correlation between x1 and y and x2 and y
  x1_corrcoeff = Formulas.coefficient_of_correlation(data["x1"], data["y"])
  x2_corrcoeff = Formulas.coefficient_of_correlation(data["x2"], data["y"])
  print(x1_corrcoeff, x2_corrcoeff)
  
  # Hypothesis testing for linearity of beta1 and beta2 at 0.01 significance level
  # Test beta1 = 0 vs beta1 != 0
  
  # Test beta2 = 0 vs beta2 != 0
  
  
  
  

# Execute main function
if __name__ == "__main__":
  main()