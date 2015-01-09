# Machine Learning Homework 9
# Exercise 13.7 of http://statweb.stanford.edu/~tibs/ElemStatLearn/
# Machine Learning Homework 9
# Exercise 13.7 Questoin 3: repeat 1 with penalty
import sys
import numpy as np
import random
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import warnings

# generate the classifier Ys for "easy" problem
def Data2Ye(data):
  Y = np.zeros((data.shape[0],), dtype = np.int)
  for i in range(data.shape[0]):
    if data[i,0]>0.5:
      Y[i] = 1
  return Y

# generate the classifier Ys for "difficulity" problem
def Data2Yd(data):
  Y = np.zeros((data.shape[0],), dtype = np.int)
  for i in range(data.shape[0]):
    sign3 = 1.0
    for j in range(3):
      sign3 = sign3 * (data[i,j] - 0.5)
    if sign3 > 0:
      Y[i] = 1
  return Y

# main() function
def main():
  print "---------- CS 574 Homework 9 -----------"
  print "  Ex. 13.7 Questions 3:\n  Repeat 1 with penalty \n"
  num_runs = 10
  num_neigs = np.arange(4, 82, 4)
  error_rates_e = np.zeros((num_neigs.shape[0], num_runs))
  error_rates_d = np.zeros((num_neigs.shape[0], num_runs))
  run = 0
  penalty = 2.0
  
  for run in range(num_runs): 
    ##### Generate Training and test dataset
    for neig in range(num_neigs.shape[0]):
      datasetX = np.random.rand(1100,10)
      trainX_e = datasetX[:100,:]
      trainX_d = datasetX[:100,:]
      testX_e = datasetX[100:,:]
      testX_d = datasetX[100:,:]
      y_e = Data2Ye(datasetX)
      y_d = Data2Yd(datasetX)
      trainY_e = y_e[:100]
      testY_e = y_e[100:]
      trainY_d = y_d[:100]
      testY_d = y_d[100:]
      
      ### train and test the data ###
      # easy problem 
      knn = KNeighborsClassifier(n_neighbors = num_neigs[neig])
      knn.fit(datasetX, y_e)
      err_rate = np.count_nonzero(knn.predict(datasetX) - y_e) / float(y_e.shape[0])
      error_rates_e[neig, run] = err_rate + 2.0/num_neigs[neig]

      # difficult problem
      knn.fit(trainX_d, trainY_d)
      err_rate = np.count_nonzero(knn.predict(datasetX) - y_e) / float(y_e.shape[0])
      error_rates_d[neig, run] = err_rate + 2.0/num_neigs[neig]
      
  mean_e = np.mean(error_rates_e, axis = 1)
  std_e = np.std(error_rates_e, axis = 1)
  mean_d = np.mean(error_rates_d, axis = 1)
  std_d = np.std(error_rates_d, axis = 1)
  
  #print "error rates for easy"
  #print std_e
  #print "error rates for diff"
  #print std_d
      #print "neig = ", neig, "error rate = ", err_rate
      #print "Data: \n", dataset
      #print "Y for easy:", Ye
      #print "Y for difficuty", Yd
  plot_x = num_neigs
  fig, (ax0, ax1) = plt.subplots(nrows = 2)
  ax0.errorbar(plot_x, mean_e, yerr = std_e/np.sqrt(std_e.shape[0]))
  ax0.set_title(' Easy problem / KNN with penalty')
  ax0.set_ylabel('Misclassification Error')
  ax0.set_xlim([4, 82])

  ax1.errorbar(plot_x, mean_d, yerr = std_d/np.sqrt(std_d.shape[0]))
  ax1.set_title(' Difficult problem / KNN with penalty')
  ax1.set_xlabel('Number of Neighbors')
  ax1.set_ylabel('Misclassification Error')
  ax1.set_xlim([4, 82])
  
  plt.savefig('knn_penalty2_30.pdf')
  plt.show()
  
  
if __name__ == '__main__':
  main()
