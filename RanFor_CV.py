# Machine Learning Project
# Random Forest with 5FCV
# 5FCV is not necessary for Ran For; 
# we use it to test the OOB error estimate 
import sys
import numpy as np
import random
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import warnings
#warnings.filterwarnings('ignore')
HIGH = 1
LOW = 0
  
def main():
  Max_Lag = 10
  repeat_num = 8
  Data_items = ["close",	"volume",	"open",	"high",	"low"]
  High_ext = "_RF_high_g_300.txt"
  Low_ext = "_RF_low_g_300.txt"
  ########################### APPL ############################
  filename = "APPL10YR.csv"
  Data_Original = np.loadtxt(filename, delimiter = ',', skiprows = 1)
  days_lag_array = np.arange(2, Max_Lag)
  error_lags_low = np.zeros((days_lag_array.size, repeat_num, 6)) 
  error_lags_high = np.zeros((days_lag_array.size,repeat_num, 6))
  # the last dimension has 6 items: train_error, train_0error, train_1error, test_error, test_0error, test_1error 
  f_h = open(filename[:4]+High_ext, 'w')
  f_l = open(filename[:4]+Low_ext, 'w')
  for i in range(days_lag_array.size):
    Days_lag = days_lag_array[i]
    Data = Data_Diff_Process(Data_Original, Days_lag)
    if i==0:
      print "\n", filename[:4], "Running"
      print "High_1 percentage: ", np.sum(Data[:,-2])/np.float(Data.shape[0])
      print "Low_1 percentage : ", np.sum(Data[:,-1])/np.float(Data.shape[0])
    print "Running Lag-Days:", Days_lag
    for j in range(repeat_num):
      ### highs ###
      error_lags_high[i, j,:] =  Five_RF_CV(Data[:,:-2], Data[:, -2])
      error_lags_low[i, j,:] =  Five_RF_CV(Data[:,:-2], Data[:, -1])
  Write2File(error_lags_high,days_lag_array, f_h)
  Write2File(error_lags_low,days_lag_array, f_l)
  
  ################################## GOOG ###########################
  filename = "GOOG10YR.csv"
  Data_Original = np.loadtxt(filename, delimiter = ',', skiprows = 1)
  days_lag_array = np.arange(2, Max_Lag)
  error_lags_low = np.zeros((days_lag_array.size, repeat_num, 6)) 
  error_lags_high = np.zeros((days_lag_array.size,repeat_num, 6))
  # the last dimension has 6 items: train_error, train_0error, train_1error, test_error, test_0error, test_1error 
  f_h = open(filename[:4]+High_ext, 'w')
  f_l = open(filename[:4]+Low_ext, 'w')
  for i in range(days_lag_array.size):
    Days_lag = days_lag_array[i]
    Data = Data_Diff_Process(Data_Original, Days_lag)
    if i==0:
      print "\n", filename[:4], "Running"
      print "High_1 percentage: ", np.sum(Data[:,-2])/np.float(Data.shape[0])
      print "Low_1 percentage : ", np.sum(Data[:,-1])/np.float(Data.shape[0])
    print "Running Lag-Days:", Days_lag
    for j in range(repeat_num):
      ### highs ###
      error_lags_high[i, j,:] =  Five_RF_CV(Data[:,:-2], Data[:, -2])
      error_lags_low[i, j,:] =  Five_RF_CV(Data[:,:-2], Data[:, -1])
  Write2File(error_lags_high,days_lag_array, f_h)
  Write2File(error_lags_low,days_lag_array, f_l)
  
  ################################## CSCO ###########################
  filename = "CSCO10YR.csv"
  Data_Original = np.loadtxt(filename, delimiter = ',', skiprows = 1)
  days_lag_array = np.arange(2, Max_Lag)
  error_lags_low = np.zeros((days_lag_array.size, repeat_num, 6)) 
  error_lags_high = np.zeros((days_lag_array.size,repeat_num, 6))
  # the last dimension has 6 items: train_error, train_0error, train_1error, test_error, test_0error, test_1error 
  f_h = open(filename[:4]+High_ext, 'w')
  f_l = open(filename[:4]+Low_ext, 'w')
  for i in range(days_lag_array.size):
    Days_lag = days_lag_array[i]
    Data = Data_Diff_Process(Data_Original, Days_lag)
    if i==0:
      print "\n", filename[:4], "Running"
      print "High_1 percentage: ", np.sum(Data[:,-2])/np.float(Data.shape[0])
      print "Low_1 percentage : ", np.sum(Data[:,-1])/np.float(Data.shape[0])
    print "Running Lag-Days:", Days_lag
    for j in range(repeat_num):
      ### highs ###
      error_lags_high[i, j,:] =  Five_RF_CV(Data[:,:-2], Data[:, -2])
      error_lags_low[i, j,:] =  Five_RF_CV(Data[:,:-2], Data[:, -1])
  Write2File(error_lags_high,days_lag_array, f_h)
  Write2File(error_lags_low,days_lag_array, f_l)
  """
  plt.figure(1, figsize=(5,5))
  tr1, = plt.plot(days_lag_array, error_lags_low[:,0]/repeat_num,'-sk',markersize=8, linewidth = 3.5, label ='Training for lows')
  te1, = plt.plot(days_lag_array, error_lags_low[:,1]/repeat_num,'--dk', markersize=8, linewidth = 3.5, label ='Test for lows')
  tr2, = plt.plot(days_lag_array, error_lags_high[:,0]/repeat_num,'-sb',markersize=8, linewidth = 3.5, label ='Training for highs')
  te2, = plt.plot(days_lag_array, error_lags_high[:,1]/repeat_num,'--db', markersize=8, linewidth = 3.5, label ='Test for highs')
  plt.grid(True)  
  plt.xlim([1.9, np.amax(days_lag_array)*1.02])
  plt.ylim([0, np.amax(np.array([np.amax(error_lags_high), np.amax(error_lags_low), 0.51*repeat_num]))/repeat_num*1.02])
  plt.xlabel('Number of lagged Days ', fontsize = 14)
  plt.ylabel('Classification error', fontsize = 14)
  #plt.legend([tr1, te1, tr2, te2],['Training for lows', 'Test for lows', 'Training for highs', 'Test for highs'],bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.)
  #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

  #plt.savefig('RanFor_diff_lags_goog.pdf')
  plt.show()
  """
def Write2File(errors, lags_array, f):
  n1_lags = errors.shape[0]
  n2_repeat = errors.shape[1]
  n3_e = errors.shape[2]
  means = np.mean(errors, axis = 1)
  stds = np.std(errors, axis = 1)
  if lags_array.size != n1_lags:
    print "\n error: Lags are different!!!!\n"
  print >> f, ("%-4s %-15s %-15s %-15s %-15s %-15s %-15s") % ("Lag", "Train", "Train1", "Train0", "Test", "Test1", "Test0")
  for i in range(n1_lags):
    print >> f, "%-4d" % (lags_array[i]),
    means = np.mean(errors[i,:,:], axis = 0)
    stds = np.std(errors[i,:,:], axis = 0)
    for j in range(n3_e):
      print >> f, ("%-7.4f") % means[j],
      print >> f, ("%-7.4f") % stds[j],
    print >> f, "\n",

  

def Five_RF_CV(X,y):
  errors = np.zeros((6,))
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
    # Random Forest
    clf = RandomForestClassifier(n_estimators=400, criterion ='gini' , max_depth=None, min_samples_split = 2, random_state = 0)
    clf.fit(Train_X, Train_y)
    Pre_Train_y = clf.predict(Train_X)
    Pre_Test_y = clf.predict(Test_X)
    train_0_wrong = 0.0
    train_1_wrong = 0.0
    for i in range(Train_X.shape[0]):
      if Train_y[i]==0 and Pre_Train_y[i]==1:
        train_0_wrong = train_0_wrong + 1.0
      elif Train_y[i]==1 and Pre_Train_y[i]==0:
        train_1_wrong = train_1_wrong + 1.0
    test_0_wrong = 0.0
    test_1_wrong = 0.0
    for i in range(Test_X.shape[0]):
      if Test_y[i]==0 and Pre_Test_y[i]==1:
        test_0_wrong = test_0_wrong + 1.0
      elif Test_y[i]==1 and Pre_Test_y[i]==0:
        test_1_wrong = test_1_wrong + 1.0    
    errors[0] = errors[0] + np.count_nonzero(Pre_Train_y - Train_y)/np.float(Train_y.size)
    errors[1] = errors[1] + train_1_wrong/np.float(np.sum(Train_y))
    errors[2] = errors[2] + train_0_wrong/(np.float(Train_y.size) - np.sum(Train_y))
    errors[3] = errors[3] + np.count_nonzero(Pre_Test_y - Test_y)/np.float(Pre_Test_y.size)
    errors[4] = errors[4] + test_1_wrong/np.float(np.sum(Test_y))
    errors[5] = errors[5] + test_0_wrong/(np.float(Test_y.size) - np.sum(Test_y))
  errors = errors/5.0
  #print "Train errors: ", errors[:3]
  #print "Test errors : ", errors[3:]
  return errors
  
  
def Data_Diff_Process(data, lag):
  newdata = np.zeros((data.shape[0] - lag, 5 * (lag-1) + 2))
  for i in range(data.shape[0] - lag):
    if data[i, -2] > data[i+1, -2]:
      newdata[i, -2] = 1
    else:
      newdata[i, -2] = 0
    if data[i, -1] > data[i+1, -1]:
      newdata[i, -1] = 0
    else:
      newdata[i, -1] = 1
    for j in range(lag-1):
      newdata[i, 5*j:5*(j+1)] = data[i+j+1, :] - data[i+j+2, :]
  means = np.mean(newdata[:,:-2], axis = 0)
  stds = np.std(newdata[:,:-2], axis = 0)
  for i in range(newdata.shape[1]-2):
    newdata[:,i] = (newdata[:,i] - means[i])/stds[i]
  return newdata
  
# Function Data_Norm(data): Normalize the predictors and target by (x-xmean)/sigma-> x*
# Input: "data" is the matrix with predictors at columns
# Output: it returns the transformed data 
def Data_Norm(data):
  means = np.mean(data, axis = 0)
  stds = np.std(data, axis = 0)
  std_data = np.zeros((data.shape[0], data.shape[1]))
  for i in range(data.shape[0]):
    for j in range(data.shape[1]):
      std_data[i,j] = (data[i,j] - means[j])/stds[j]
  return std_data


# Function Data_Reorg(data): reorganize the time series data to standard format: (x1,x2,..., xp; y)
# Input: 'data' is the time series matrix with data at the same day on the same row
#        'days' is a integer number of previous days to include in the model
# Output: it returns the transformed data 
def Data_Reorg(data, lag, high_low):
  org_data = np.zeros((data.shape[0] - lag, 5 * lag + 1))
  for i in range(data.shape[0] - lag):
    org_data[i, lag*5] = data[i, -1-high_low]
    for j in range(5):
      for k in range(lag):
        org_data[i, k*5 + j] = data[i+k+1, j]
  return org_data


# Function Data_Logistic(data, oneor0): convert the continuous y to 0 and 1
# It checks the data(3,:) to compare highs and data(4,:) to compare lows,
# so data(:,3) is supposed to store the highs of the previous day; data(:,4) is lows of the previous day
# Input: "data" is the matrix with predictors at column data(:-2) and target at data(-2:)
#        'oneor0' is either 1 or 0. 0 is for lows; 1 is for highs
# Output: it returns the transformed data with 0 and 1's replaced at data(:,-1) 
def Data_Logistic(data, oneor0):
  org_data = np.zeros((data.shape[0], data.shape[1]-1))
  org_data[:,:-1] = data[:,:-2]
  for i in range(data.shape[0]):
    if data[i, -1-oneor0] >= data[i, 4-oneor0]:
      org_data[i,-1] = oneor0
    else:
      org_data[i, -1] = 1 - oneor0
  return org_data
  
  
def Data_Diff(data, Days_lag):
  data_diff = np.zeros((data.shape[0], data.shape[1]-5))
  data_diff[:,-2:] = data[:,-2:]
  for i in range(Days_lag-1):
    data_diff[:, i*5:(i+1)*5] = data[:, i*5:(i+1)*5] - data[:, (i+1)*5:(i+2)*5]
  return data_diff



if __name__ == '__main__':
  main()

