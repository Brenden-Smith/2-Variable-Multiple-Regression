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

# Test the hypothesis of linearity b = 0 vs b != 0 at significance level alpha = 0.01
def test_hypothesis(b, se, n):
  print(b, se, n)
  
  # Calculate the z-statistic 
  t_stat = (b - 0) / se
  
  # Calculate the p-value
  p_value = stats.t.sf(np.abs(t_stat), n - 1) * 2
  print("p-value:", p_value)
  
  # If the p-value is less than 0.01, reject the null hypothesis. Else, accept the null hypothesis.
  if p_value < 0.01: return False
  else: return True

# Main function
def main():
  
  # Import data from csv file using pandas
  data = pd.read_csv("data.csv")
  
  # Plot data points on a 3D graph using matplotlib
  fig = plt.figure()
  ax= fig.add_subplot(111, projection='3d')
  ax.scatter(data["x1"], data["x2"], data["y"])
  
  # Label the axes
  ax.set_xlabel("Critic Score")
  ax.set_ylabel("User Scores")
  ax.set_zlabel("Album Sales")
  
  # Create X and Y matrices
  X_mat = np.column_stack(([1] * len(data["x1"]), data["x1"], data["x2"]))
  Y_mat = np.reshape(np.array(data["y"]), (len(data["y"]), 1))
  
  # Calculate beta0, beta1, and beta2
  beta0, beta1, beta2 = Formulas.beta(X_mat, Y_mat)
  print("beta0:", beta0)
  print("beta1:", beta1)
  print("beta2:", beta2)
  
  # Calculate the regression values for y
  Y_reg = []
  for i in range(len(data["x1"])):
    Y_reg.append(Formulas.regression_line(data["x1"][i], data["x2"][i], beta0, beta1, beta2))
  
  # Calculate SSE vector
  sse = Formulas.sse(data["y"], Y_reg)
  print("SSE:", sse)
  
  # Calculate MSE
  mse = Formulas.mse(len(data["y"]), sse)
  print("MSE:", mse)
  
  # Calculate the standard errors of beta0, beta1, and beta2
  se_beta0, se_beta1, se_beta2 = Formulas.se_beta(mse, X_mat)
  print("se_beta0:", se_beta0)
  print("se_beta1:", se_beta1)
  print("se_beta2:", se_beta2)
  
  # Calculate the coefficient of correlation between x1 and y and x2 and y
  x1_corrcoeff = Formulas.coefficient_of_correlation(data["x1"], data["y"])
  x2_corrcoeff = Formulas.coefficient_of_correlation(data["x2"], data["y"])
  print("x1_corrcoeff:", x1_corrcoeff)
  print("x2_corrcoeff:", x2_corrcoeff)
  
  # Hypothesis testing for linearity of beta1 and beta2 at 0.01 significance level
  # Test beta1 = 0 vs beta1 != 0
  if(test_hypothesis(beta1, se_beta1, len(data["y"]))): print("Reject the null hypothesis that beta1 = 0.")
  else: print("Accept the null hypothesis that beta1 = 0.")
  
  # Test beta2 = 0 vs beta2 != 0
  if(test_hypothesis(beta2, se_beta2, len(data["y"]))): print("Reject the null hypothesis that beta2 = 0.")
  else: print("Accept the null hypothesis that beta2 = 0.")
  
  # Plot the lines of best fit
  sorted_x1 = sorted(data["x1"])
  sorted_x2 = sorted(data["x2"])
  sorted_y_reg = []
  for i in range(len(sorted_x1)):
    sorted_y_reg.append(Formulas.regression_line(sorted_x1[i], sorted_x2[i], beta0, beta1, beta2))
  plt.plot(sorted_x1, sorted_x2, sorted_y_reg, label="Regression Line", color="red")
  
  
  # Save graph as gif
  for ii in np.arange(0, 360, 1):
    ax.view_init(elev=32, azim=ii)
    fig.savefig('./gif/gif_image%d.png' % ii)
    
    # Show the graph
  plt.show()
  
  

# Execute main function
if __name__ == "__main__":
  main()