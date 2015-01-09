# Data Autocorrelation function & cross-feature correlation
# This code is to analyze the autocorrelation & cross-feature correlation matrix
# of stock prices data. The data files can be downloaded from NASDAQ official website

import sys
import numpy as np
import pandas as pd
from pandas.tools.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
import warnings

def main():
  ############## Autocorrelation function of original-standarized data #################
  filename = "APPL10YR_t.csv"
  A = pd.read_csv(filename)
  Data = Read_DF2Array(A)

  Data_n = Data_Norm(Data)
  N_row = Data.shape[0]
  N_col = Data.shape[1]
  print "###################  APPL  ##################"
  f = open("Correlation_Matrix2.txt",'w')
  print >> f, "###################  APPL  ##################"
  Data_items = ["close",	"volume",	"open",	"high",	"low"]
  Cor_Mat=np.corrcoef(Data_n, rowvar = 0)
  print "Correlation Matrix of original data(standardized ):\n",
  print >>f, "Correlation Matrix of original data(standardized ):\n",
  Print_Cor_Mat(Cor_Mat, Data_items)
  Print_Cor_Mat_File(Cor_Mat, Data_items, f)
  Data = Data_Diff(Data)
  Data_n = Data_Norm(Data)
  Cor_Mat_Diff=np.corrcoef(Data_n, rowvar = 0)
  print "Correlation Matrix of day-by-day change data(standardized):\n",
  print >> f, "Correlation Matrix of day-by-day change data(standardized):\n",
  Print_Cor_Mat(Cor_Mat_Diff, Data_items)
  Print_Cor_Mat_File(Cor_Mat_Diff, Data_items, f)
  print "\n"
  print >>f, "\n"
  
########### google ############
  filename = "GOOG10YR_t.csv"
  A = pd.read_csv(filename)
  Data = Read_DF2Array(A)

  Data_n = Data_Norm(Data)
  N_row = Data.shape[0]
  N_col = Data.shape[1]
  Data_items = ["close",	"volume",	"open",	"high",	"low"]
  Cor_Mat=np.corrcoef(Data_n, rowvar = 0)
  print "###################  GOOG  ##################"
  print >> f, "###################  GOOG  ##################"
  print "Correlation Matrix of original data(standardized ):\n",
  print >>f, "Correlation Matrix of original data(standardized ):\n",
  Print_Cor_Mat(Cor_Mat, Data_items)
  Print_Cor_Mat_File(Cor_Mat, Data_items, f)
  Data = Data_Diff(Data)
  Data_n = Data_Norm(Data)
  Cor_Mat_Diff=np.corrcoef(Data_n, rowvar = 0)
  print "Correlation Matrix of day-by-day change data(standardized):\n",
  print >> f, "Correlation Matrix of day-by-day change data(standardized):\n",
  Print_Cor_Mat(Cor_Mat_Diff, Data_items)
  Print_Cor_Mat_File(Cor_Mat_Diff, Data_items, f)
  print "\n"
  print >>f, "\n"


########### CSCO ############
  filename = "CSCO10YR_t.csv"
  A = pd.read_csv(filename)
  Data = Read_DF2Array(A)
  Data_n = Data_Norm(Data)
  N_row = Data.shape[0]
  N_col = Data.shape[1]
  Data_items = ["close",	"volume",	"open",	"high",	"low"]
  Cor_Mat=np.corrcoef(Data_n, rowvar = 0)
  print "###################  CSCO  ##################"
  print >> f, "###################  CSCO  ##################"
  print "Correlation Matrix of original data(standardized ):\n",
  print >> f, "Correlation Matrix of original data(standardized ):\n",
  Print_Cor_Mat(Cor_Mat, Data_items)
  Print_Cor_Mat_File(Cor_Mat, Data_items, f)
  Data = Data_Diff(Data)
  Data_n = Data_Norm(Data)
  Cor_Mat_Diff=np.corrcoef(Data_n, rowvar = 0)
  print "Correlation Matrix of day-by-day change data(standardized):\n",
  print >> f, "Correlation Matrix of day-by-day change data(standardized):\n",
  Print_Cor_Mat(Cor_Mat_Diff, Data_items)
  Print_Cor_Mat_File(Cor_Mat_Diff, Data_items, f)
  print >>f, "\n"


#print correlation matrix to file 
def Print_Cor_Mat_File(cm, items, f):
  n = cm.shape[0]
  f.write("         ")
  for i in range(n):
    print >> f, ("%-9s") % (items[i]),
  f.write("\n")
  for i in range(n):
    print >> f, "%-9s" % (items[i]),
    for j in range(n):
      print >>f, "%-9.4f" % (cm[i][j]),
    f.write("\n")
  f.write("\n\n")

    
 
#print correlation matrix to screen 
def Print_Cor_Mat(cm, items):
  n = cm.shape[0]
  print "         ",
  for i in range(n):
    print "%-9s" % (items[i]),
  print "\n",
  for i in range(n):
    print "%-9s" % (items[i]),
    for j in range(n):
      print "%-9.4f" % (cm[i][j]),
    print "\n",
  print "\n"
    
    
# this function is only for specific data struction
# close	volume	open	high	low
def Read_DF2Array(A):
  Data = np.zeros((A.high.size, 5))
  Data[:,0] = (A.close.values).transpose()
  Data[:,1] = (A.volume.values).transpose()
  Data[:,2] = (A.open.values).transpose()  
  Data[:,3] = (A.high.values).transpose()
  Data[:,4] = (A.low.values).transpose()
  return Data

def Autocor(x):
    n = len(x)
    acf = np.zeros((n,))
    for i in range(n-1):
      m1 = np.mean(x)
      m2 = np.mean(x)
      acf[i] = 1.0/(n)*np.sum((x[:n-i]-m1)*(x[i:]-m2))/np.std(x)**2
    return acf
    
def Data_Diff(data):
  data_diff = np.zeros((data.shape[0]-1, data.shape[1]))
  for i in range(data.shape[0]-1):
    data_diff[i,: ] = data[i,:]  - data[i+1,:]
  return data_diff
    
def Data_Norm(data):
  price_data = np.zeros((data.shape[0]*4))
  N = data.shape[0]
  for i in range(4):
    price_data[i*N:(i+1)*N] = data[:,i]
  price_data[N:2*N] = data[:,4]
  means = np.mean(data, axis = 0)
  means[0] = np.mean(price_data)
  means[2] = means[0]
  means[3] = means[0]
  means[4] = means[0]
  stds = np.std(data, axis = 0)
  stds[0] = np.std(price_data)
  stds[2] = stds[0]  
  stds[3] = stds[0]
  stds[4] = stds[0]
  std_data = np.zeros((data.shape[0], data.shape[1]))
  for i in range(data.shape[0]):
    for j in range(data.shape[1]):
      std_data[i,j] = (data[i,j] - means[j])/stds[j]
  return std_data


if __name__ == '__main__':
  main()
