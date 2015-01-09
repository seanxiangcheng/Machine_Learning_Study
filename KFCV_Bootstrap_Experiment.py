# CS574 Homework 7
# 5-fold cross validation & bootstrapping
# It was the my first Python-Machine Learning code which also made me like Python more than Matlab:) 
# by Xiang Cheng

import sys
import numpy as np
import random
from sklearn import linear_model as LM
import warnings
warnings.filterwarnings('ignore')
# there are warning because of log(0.0) which is out the precision of python

# function to return the value of the vaule of a item in a dictionary
# just for the function "sorted()"
def get_count(wd_tup):
  return wd_tup[1]

# function to read the data file
# Input filename: make sure the file is in text format and in the current directionary
# Output: returns a Data list with class number in the last column; 
# the class are named 0, 1, 2... according to their sequence in the file
def Read_Data(filename):
  f = open(filename, 'r')
  data = [] # data will be stored here as a N*5 2D list
  class_dict = {}
  class_num = 0  # the class will be numbered sequentially
  row = 0
  for line in f:
    data.append([])
    linesplit = line.split(',')
    for nw in linesplit[0:4]:
      data[row].append(float(nw))
    class_name = linesplit[4].strip()
    if not (class_name in class_dict):
      class_dict[class_name] = class_num
      class_num += 1
    data[row].append(class_dict[class_name])
    row += 1
  print '       The classes are named as follows:'
  print '       Class_Name         Class_Number'
  class_items = sorted(class_dict.items(), key = get_count)
  for word in class_items:
    print "       %-16s  :    %-3d " % (word[0], word[1])
  return data

def error_calc(z, y):
  error = 0
  row = 0
  for i in y:
    error = error - z[row, i]
    row = row + 1
  error = error*2.0/len(y)
  return error

# 5-fold Cross Validation
# Input: features X in array, each row is one sample
#        Classes in array, each element is a corresponding class of X
# Output: returns the generalization error
def Five_CV(X,y):
  error = 0.0
  test_error_rate = 0.0
  ### First generate the index of the 5 subsets
  row_index = range(X.shape[0])
  subsets_indices = np.zeros((5, X.shape[0]/5),dtype=np.int) # indices of each section is stored in one row
  for i in range(5):
    for j in range(X.shape[0]/5):
      subsets_indices[i,j] = random.choice(row_index)
      row_index.remove(subsets_indices[i,j])
 
  ### Create the subsets for Trainning and Test and record the error separtely
  Train_X = np.zeros((4*X.shape[0]/5, X.shape[1]))
  Train_y = np.zeros((4*X.shape[0]/5, ), dtype = np.int)
  Test_X = np.zeros((X.shape[0]/5, X.shape[1]))
  Test_y = np.zeros((X.shape[0]/5, ), dtype = np.int)

  for k in range(5):
    # create subsets for trainning and test
    rows = range(5)
    rows.remove(k)
    count = 0
    for i in rows:
      for j in range(subsets_indices.shape[1]):
        Train_X[count*X.shape[0]/5+j,:] = X[subsets_indices[i,j],:]
        Train_y[count*X.shape[0]/5+j] = y[subsets_indices[i,j]]
      count = count+1
    Train_y = Train_y.astype(int)
    for j in range(subsets_indices.shape[1]):
      Test_X[j,:] = X[subsets_indices[k,j],:]
      Test_y[j] = y[subsets_indices[k,j]]
    Test_y = Test_y.astype(int)
    # logistic regressions
    logreg = LM.LogisticRegression(C=1e8) # Large C means there is almost no L1 or L2 penalty
    logreg.fit(Train_X,Train_y)
    z = logreg.predict_log_proba(Test_X)
    error = error + error_calc(z, Test_y)
    test_error_rate = test_error_rate + np.count_nonzero(logreg.predict(Test_X)-Test_y)/float(len(Test_y))
  print "            Average Test classification error of Cross Validation: %-.4f" % (test_error_rate/5.0)
  return error/5.0
  
  

# Bootstrapping Cross Validation
# Input: features X in array, each row is one sample
#        Classes in array, each element is a corresponding class of X
# Output: returns the generalization error
def Bts_CV(X, y, B):
  error = 0.0   # generalized error using sum(log P)
  c_error = 0.0 # classification error 
  train_error = 0.0
  test_error = 0.0
  train_error_rate = 0.0
  test_error_rate = 0.0
  num_selected = 0
  N = len(y)
  all_rows = range(N)
  for b in range(B):
    # set up the bootstrap training and test samples
    rows_selected = []
    rows_left = range(N)
    for i in range(N):
      r = random.choice(all_rows)
      rows_selected.append(r)
      if r in rows_left:
        rows_left.remove(r)
    num_unselected = len(rows_left)
    num_selected = num_selected + N - num_unselected
    Train_X = np.zeros((N,X.shape[1]))
    Train_y = np.zeros((N,), dtype =np.int)
    Test_X = np.zeros((num_unselected,X.shape[1]))
    Test_y = np.zeros((num_unselected,), dtype = np.int)
    for i in range(N):
      Train_X[i,:] = X[rows_selected[i],:]
      Train_y[i] = y[rows_selected[i]]
    for i in range(num_unselected):
      Test_X[i,:] = X[rows_left[i],:]
      Test_y[i] = y[rows_left[i]]
    
    # Train the data
    logreg = LM.LogisticRegression(C=1e8) # Large C means there is almost no L1 or L2 penalty
    logreg.fit(Train_X,Train_y)
    z = logreg.predict_log_proba(Train_X)
    train_error = train_error + error_calc(z, Train_y)*N
    train_error_rate = train_error_rate + np.count_nonzero(logreg.predict(Train_X)-Train_y)/float(len(Train_y))
    
    z = logreg.predict_log_proba(Test_X)
    test_error = test_error + error_calc(z, Test_y)*num_unselected
    test_error_rate = test_error_rate + np.count_nonzero(logreg.predict(Test_X)-Test_y)/float(num_unselected)
  
  error = 0.368*train_error/num_selected + 0.632*test_error/(N*B-num_selected)
  test_error_rate = test_error_rate / B
  c_error_rate = 0.368*train_error_rate/B + 0.632*test_error_rate
  print '            Total number of bootstraps: %d' % (B)
  print '            Average ratio of unique samples: %.2f/%d = %-.4f' % (num_selected/float(B), N, num_selected/float(B)/float(N))
  print '            Test error rate: %-.4f' % (test_error_rate)
  print '            Generalized classification error rate: %-.4f' % (c_error_rate)
  return error






def main():
  print '  CS 574 Homewoerk 7, by Xiang Cheng \n'
  print '  Two Ways to use this program:'
  print '  Option 1: \n    use Terminal, cd to the directionary; \n    copy data fle to the directionary; \n    type "python HW7XC.py DataFileName "'
  print '  Option 2: \n    save the data to text file in the same directionary; \n    name the data file \'data\'\n\n'
  
  data_filename='data'
  if len(sys.argv)>1:
    data_filename=sys.argv[1]

  print '   *********** Program running **************'
  # read data file and convert Iris-**** to class numbers: 1, 2, 3 ...
  # returns a matrix with data and class number (last column)
  print '       Reading data file: \'%s\'  ......\n' % (data_filename) 
  data = Read_Data(data_filename);
  data = np.asarray(data)
  X = data[:,:4]  # Split featues and classes to X and Y, respectively
  y = data[:,4].astype(int)
  # Create a Logistic Regression
  logreg = LM.LogisticRegression(C=1e8) # Large C means there is almost no L1 or L2 penalty
  logreg.fit(X,y)
  z = logreg.predict(X)
  
  print  '\n        *** Test: Logi Reg with all data ***' 
  print '            Error rate: %-.4f' % (np.count_nonzero(z-y)/float(len(z)))
  z = logreg.predict_log_proba(X);
  print '            Error = %-.4f' % (error_calc(z,y))
  
  print '\n    *************  Validation Running **************'
  #########5-fold Cross-Validation ###########
  print '\n        *** 5-fold Cross-Validation Running... ***'
  g_error_5fcd = Five_CV(X, y)
  print '            Generalized error = %-.4f ' % g_error_5fcd
  
  ######### Bootstrapping ###########
  print '\n        *** .632 Bootstrapping Running ...***'
  B = 500  # number of repetitions of bootstrapping
  g_error_bts = Bts_CV(X, y, B)
  print '            Generalized error = %-.4f \n' % (g_error_bts)
  
  print '  ******** Code Running Done for Xiang Cheng\'s Homework 7 ********'
  
  
if __name__ == '__main__':
  main()
