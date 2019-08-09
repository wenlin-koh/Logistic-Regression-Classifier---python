#==============================================================================
# title           : logistic_reg.py
# author          : Koh Wen Lin
# date            : 3/06/2019
# description     : This file contains the implementations of logistic 
#                   regression for cs385 assignment 2.
# python_version  : 3.7.3  
#==============================================================================
import math
import numpy as np
from scipy import special
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

#==============================================================================
# Brief : Function takes in a file name and read the data set into an array.
#
# Parameter :
#    filename (String) : File name of the file to be read.
# 
# Returns :
#    A 2D Array containing the data set.
#==============================================================================
def dataLoad(filename):
  # Stores the data in filename.
  X = []
  count = 0

  dataFile = open(filename, "r")
  lines = dataFile.readlines()

  # Read in line and store in list to append into X for each line
  for line in lines:
    X.append([])
    words = line.split(",")
    for word in words:
      X[count].append(float(word))
    count += 1
  
  dataFile.close()
  return np.asarray(X)

def dataSave(filename, x_set):
  # Save data in x into filename

  dataFile = open(filename, "w")

  for x in x_set:
    for i in range(x.shape[0]):
      dataFile.write(str(x[i]))

      if i < x.shape[0] - 1:
        dataFile.write(",")
      else:
        dataFile.write("\n")
  
  dataFile.close()

#==============================================================================
# Brief : Function normalizes the input data set.
#
# Parameter :
#    X (2D array) : Data set to be normalized
# 
# Returns :
#    A 2D Array containing the normalized data set.
#==============================================================================
def dataNorm(X):
  iLen = X.shape[0]
  jLen = X.shape[1]

  # Stores the normalized dataset
  X_norm = []

  maxX = np.full((jLen), -float("inf"))
  minX = np.full((jLen), float("inf"))

  # Build the unnormalized data set and find min and max for all attribute
  for i in range(iLen):
    maxX = np.maximum(X[i], maxX)
    minX = np.minimum(X[i], minX)
    
  # normalized the data set
  for i in range(iLen):
    X_norm.append([])
    X_norm[i] = (X[i] - minX) / (maxX - minX)
    X_norm[i] = np.insert(X_norm[i], 0, 1.0, axis = 0)

  return np.asarray(X_norm)

#==============================================================================
# Brief : Function prints the mean and sum of the normalized dataset for 
#         correctness checking.
#
# Parameter :
#    X_norm (list) : List of normalized data set to check correctness
#==============================================================================
def testNorm(X_norm):
  xMerged = np.copy(X_norm[0])

  for i in range(len(X_norm ) - 1):
    xMerge = np.concatenate((xMerged, X_norm[i + 1]))
  
  X_mean = np.mean(xMerged, axis = 0)
  X_sum = np.sum(xMerged, axis = 0)
  print([round(n, 4) for n in X_mean])
  print([round(n, 3) for n in X_sum])

#==============================================================================
# Brief : Function computes the cost/error function of the dataset.
#
# Parameter :
#    X_norm (2D array (1372, 6)) : Array containing the normalized dataset.
#
#    Theta (2D array (5, 1)) : Array containing the coefficients vector.
#
# Returns :
#    The cost of the dataset with given coefficient vector.
#==============================================================================
def errCompute(X_norm, Theta):
  cost = 0.0

  # Summation of the cost function
  for x in X_norm:
    x_input = x[:-1]
    y_actual = x[-1]
    yHat = np.dot(Theta.reshape(5), x_input)
    yHat_w = special.expit(yHat)
    cost += y_actual * math.log(yHat_w) + (1 - y_actual) * math.log(1 - yHat_w)

  # Error function
  return (-1 / X_norm.shape[0]) * cost

#==============================================================================
# Brief : Function evaluates Theta using Stochastic Gradient Descent algorithm.
#
# Parameter :
#    X (2D array (#, 6)) : Array containing the training and predicting 
#                          dataset.
#
#    Theta (2D array (5, 1)) : Array containing the coefficients vector.
#
#    Alpha (float) : Learning rate.
#
#    Num_iters (int) : Number of iteration to run.
#
# Returns :
#    The learned parameter Theta.
#==============================================================================
def stochasticGD(X, Theta, Alpha, Num_iters):
  coeff = np.copy(Theta)
  num_epochs = int(Num_iters / X.shape[0])

  iteration = 0
  errList = [[errCompute(X, coeff), iteration]]

  # Run the Stochastic Gradient Descent algorithm to get the best coefficient factors
  for i in range(num_epochs):
    converged = True
    for x in X:
      yHat = special.expit(np.dot(x[:-1], coeff.reshape(5)))
      y = x[-1]
      error = y - yHat
      
      # update parameters if predicted output is different from actual output
      if abs(error) > 0:
        converged = False
        for j in range(coeff.shape[0]):
          coeff[j] = coeff[j] + Alpha * error * x[j]

      iteration += 1
      errList.append([errCompute(X, coeff), iteration])

    # Repeat epochs until converged
    if converged:
      break

  return coeff, errList

#==============================================================================
# Brief : Function performs the prediction of dataset output using given
#         Theta(weights).
#
# Parameter :
#    X_norm (2D array (#, 6)) : Array containing the predicting dataset.
#
#    Theta (2D array (5, 1)) : Array containing the coefficients vector.
#
# Returns :
#    An array containing the predicted output value.
#==============================================================================
def Predict(X_norm, Theta):
  y = []
  for x in X_norm:
    y.append(logisticFunction(x[:-1], Theta.reshape(5)))
  
  return np.asarray(y)

#==============================================================================
# Brief : Function performs logistic function operation on given parameters and
#         attributes.
#
# Parameter :
#    X (1D array (#)) : Array containing the attribute vector.
#
#    Theta (1D array (#)) : Array containing the parameters vector.
#
# Returns :
#    1 or 0.
#==============================================================================
def logisticFunction(X, Theta):
  yHat = special.expit(np.dot(Theta, X))
  return 1 if yHat >= 0.5 else 0


#==============================================================================
# Brief : Function splits given dataset into training and testing set based on
#         percentTT (Training).
#
# Parameter :
#    X_norm (2D array (#, 6)) : Array containing the dataset to split.
#
#    percentTT (float) : Percentage to split for training.
#
# Returns :
#    A list containing a training dataset and a testing dataset.
#==============================================================================
def splitTT(X_norm, percentTT):
  pivot = int(X_norm.shape[0] * percentTT)
  X_train = X_norm[:pivot]
  X_test = X_norm[pivot:]
  
  return [X_train, X_test]

def testPrediction(X_test, Theta):
  y_predict = Predict(X_test, Theta)
  
  hit = 0
  for i in range(X_test.shape[0]):
    if X_test[i][-1] == y_predict[i]:
      hit += 1

  return hit / X_test.shape[0] * 100.0, y_predict, np.asarray([int(n[-1]) for n in X_test[:]])

def countTPFPTNFN(y_actual, y_predict):
  TP = 0
  FP = 0
  TN = 0
  FN = 0

  for i in range(y_predict.shape[0]):
    if y_actual[i] == y_predict[i]:
      if y_predict[i] == 1:
        TP += 1
      else:
        TN += 1
    else:
      if y_predict[i] == 1:
        FP += 1
      else:
        FN += 1
  
  return TP, FP, TN, FN

def outputSplitSet():
  X = dataLoad("data_banknote_authentication.txt")
  X_set = []

  # duplicate 5 dataset and shuffle them
  for i in range(5):
    X_set.append(np.copy(X))
    np.random.shuffle(X_set[i])

  i = 1
  for shuffled_set in X_set:
    dataSave("a2shuffled_set{}.data".format(i), shuffled_set)
    split = splitTT(shuffled_set, 0.6)
    dataSave("a2shuffled_train_set{}.data".format(i), split[0])
    dataSave("a2shuffled_test_set{}.data".format(i), split[1])
    i += 1

def outputSplitSet2():
  X_set = []

  for i in range(5):
    X_set.append(dataLoad("a2shuffled_set{}.data".format(i + 1)))

  i = 1
  for shuffled_set in X_set:
    split = splitTT(shuffled_set, 0.6)
    dataSave("a2shuffled_train_set{}.data".format(i), split[0])
    dataSave("a2shuffled_test_set{}.data".format(i), split[1])
    i += 1

def test0():
  print("Running " + test0.__name__)
  print("Train set : a4shuffled.data")
  X = dataLoad("a4shuffled.data")
  X_shufnorm = dataNorm(X)
  theta = stochasticGD(X_shufnorm, np.zeros((X_shufnorm.shape[1] - 1, 1)), 0.01, 1372 * 20)
  J = errCompute(X_shufnorm, theta)
  print("Error : {}".format(J))
  
  X_test = dataLoad("data_banknote_authentication.txt")
  X_testnorm = dataNorm(X_test)

  print("Test set : data_banknote_authentication.txt")
  y_predict = Predict(X_testnorm, theta)

  hit = 0
  for i in range(X_testnorm.shape[0]):
    if X_testnorm[i][-1] == y_predict[i]:
      hit += 1
  accuracy = hit / X_testnorm.shape[0] * 100.0
  print("Accuracy : {}".format(accuracy))

def test1():
  print("Running " + test1.__name__)
  X = dataLoad("data_banknote_authentication.txt")
  X_norm = dataNorm(X)
  X_normset = []

  # duplicate 5 dataset and shuffle them
  for i in range(5):
    X_normset.append(np.copy(X_norm))
    np.random.shuffle(X_normset[i])
  
  # split all 5 set to train-and-test split 60,40
  X_splitset = [splitTT(n, 0.6) for n in X_normset]
  
  X_thetaset = [stochasticGD(x_split[0], np.zeros((x_split[0].shape[1] - 1, 1)), 0.01, x_split[0].shape[0] * 20) for x_split in X_splitset]

  X_accuracyset = []
  
  id = 0
  for x_split in X_splitset:
    X_accuracyset.append(testPrediction(x_split[1], X_thetaset[id]))
    id += 1
  
  print(X_accuracyset)

def learningRateTest(learning_rate):
  print("Running " + learningRateTest.__name__ + " with learning rate = {}".format(learning_rate))
  X_set = []

  for i in range(5):
    X_set.append(dataLoad("a2shuffled_set{}.data".format(i + 1)))
  
  X_normset = [dataNorm(X) for X in X_set]
  
  # split all 5 set to train-and-test split 60,40
  X_splitset = [splitTT(n, 0.6) for n in X_normset]

  X_thetaset = [stochasticGD(x_split[0], np.zeros((x_split[0].shape[1] - 1, 1)), learning_rate, x_split[0].shape[0] * 20) for x_split in X_splitset]

  X_accuracyset = []
  
  id = 0
  for x_split in X_splitset:
    accuracy, y_predict = testPrediction(x_split[1], X_thetaset[id])
    X_accuracyset.append(accuracy)
    id += 1
  
  print([round(n, 4) for n in X_accuracyset])

def plotROCCurves():
  print("Running " + plotROCCurves.__name__)
  X_set = []
  lr = 1.0

  for i in range(5):
    X_set.append(dataLoad("a2shuffled_set{}.data".format(i + 1)))
  
  X_normset = [dataNorm(X) for X in X_set]
  
  # split all 5 set to train-and-test split 60,40
  X_splitset = [splitTT(n, 0.6) for n in X_normset]

  X_thetaset = [stochasticGD(x_split[0], np.zeros((x_split[0].shape[1] - 1, 1)), lr, x_split[0].shape[0] * 20) for x_split in X_splitset]

  X_accuracyset = []
  X_predictset = []
  X_outputset = []
  rocaucset = []
  fprset = []
  tprset = []

  id = 0
  for x_split in X_splitset:
    accuracy, y_predict, y_actual = testPrediction(x_split[1], X_thetaset[id])
    X_accuracyset.append(accuracy)
    X_predictset.append(y_predict)
    X_outputset.append(y_actual)
    rocauc = roc_auc_score(y_actual, y_predict)
    rocaucset.append(rocauc)
    fpr, tpr, thresholds = roc_curve(y_actual, y_predict)
    fprset.append(fpr)
    tprset.append(tpr)
    id += 1

  plt.figure()
  plt.title('Receiver Operating Characteristic')
  plt.plot(fprset[0], tprset[0], color = 'red', lw = 2, label = 'Logistic Regression on a2shuffled_set1.data (area = {}'.format(round(rocaucset[0], 2)))
  plt.plot(fprset[1], tprset[1], color = 'blue', lw = 2, label = 'Logistic Regression on a2shuffled_set2.data (area = {}'.format(round(rocaucset[1])))
  plt.plot(fprset[2], tprset[2], color = 'green', lw = 2, label = 'Logistic Regression on a2shuffled_set3.data (area = {}'.format(round(rocaucset[2])))
  plt.plot(fprset[3], tprset[3], color = 'yellow', lw = 2, label = 'Logistic Regression on a2shuffled_set4.data (area = {}'.format(round(rocaucset[3])))
  plt.plot(fprset[4], tprset[4], color = 'purple', lw = 2, label = 'Logistic Regression on a2shuffled_set5.data (area = {}'.format(round(rocaucset[4])))
  plt.plot([0,1], [0,1], 'r--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.legend(loc='lower right')
  plt.show()
  
  print([round(n, 4) for n in X_accuracyset])

def plotROCCurve(filename, color = 'red'):
  print("Running " + plotROCCurves.__name__ + " with {}".format(filename))

  lr = 1.0
  num_epochs = 20

  X = dataLoad(filename)
  X_norm = dataNorm(X)
  X_split = splitTT(X_norm, 0.6)
  X_theta = stochasticGD(X_split[0], np.zeros((X_split[0].shape[1] - 1, 1)), lr, X_split[0].shape[0] * num_epochs)
  
  errJ = errCompute(X_split[0], X_theta)
  print("Error function : {}".format(errJ))
  
  print("Learned Parameters : w0 = {}, w1 = {}, w2 = {}, w3 = {}, w4 = {}".format(X_theta[0], X_theta[1], X_theta[2], X_theta[3], X_theta[4]))
  accuracy, y_predict, y_actual = testPrediction(X_split[1], X_theta)
  print(round(accuracy, 4))

  fpr, tpr, thresholds = roc_curve(y_actual, y_predict)
  roc_auc = roc_auc_score(y_actual, y_predict)

  con_matrix = confusion_matrix(y_actual, y_predict)
  print(con_matrix)

  plt.figure(figsize=(16,9))
  plt.title('Receiver Operating Characteristic')
  # plt.plot(errList[:][-1], errList[:][0], color)
  plt.plot(fpr, tpr, color, lw = 2, label = 'Logistic Regression on a2shuffled_set1.data (area = {})'.format(round(roc_auc, 2)))
  plt.plot([0,1], [0,1], 'r--')
  plt.xlim([0, 1.0])
  plt.ylim([0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.legend(loc='lower right')
  plt.show()

def plotError(filename, color = 'red'):
  print("Running " + plotError.__name__ + " with {}".format(filename))

  lr = 1.0
  num_epochs = 20

  X = dataLoad(filename)
  X_norm = dataNorm(X)
  X_split = splitTT(X_norm, 0.6)
  X_theta, errList = stochasticGD(X_split[0], np.zeros((X_split[0].shape[1] - 1, 1)), lr, X_split[0].shape[0] * num_epochs)
  
  errJ = errCompute(X_split[0], X_theta)
  print("Error function : {}".format(errJ))
  
  print("Learned Parameters : w0 = {}, w1 = {}, w2 = {}, w3 = {}, w4 = {}".format(X_theta[0], X_theta[1], X_theta[2], X_theta[3], X_theta[4]))
  accuracy, y_predict, y_actual = testPrediction(X_split[1], X_theta)
  print(round(accuracy, 4))

  plt.figure(figsize=(16,9))
  plt.plot(errList[:][-1], errList[:][0], color)
  plt.xlabel('Number of Iteration')
  plt.ylabel('Error')
  plt.show()

#def plotROCCurve2(filename, color = 'red'):
#  print("Running " + plotROCCurves.__name__ + " with {}".format(filename))
#
#  lr = 1.0
#  num_epochs = 20
#
#  X = dataLoad(filename)
#  X_norm = dataNorm(X)
#  X_split = splitTT(X_norm, 0.6)
#  logreg = LogisticRegression()
#  train = X_split[0][:,:-1]
#  logreg.fit(X_split[0][:,:-1], X_split[0][:,-1])
#  y_pred = logreg.predict(X_split[1][:,:-1])
#  acc = logreg.score(X_split[1][:,:-1], X_split[1][:,-1])
#
#  fpr, tpr, thresholds = roc_curve(X_split[1][:,-1], y_pred)
#  roc_auc = roc_auc_score(X_split[1][:,-1], y_pred)
#
#  plt.figure()
#  plt.title('Receiver Operating Characteristic')
#  plt.plot(fpr, tpr, color, lw = 2, label = 'Logistic Regression on a2shuffled_set1.data (area = {})'.format(round(roc_auc, 2)))
#  plt.plot([0,1], [0,1], 'r--')
#  plt.xlim([0.0, 1.0])
#  plt.ylim([0.0, 1.05])
#  plt.xlabel('False Positive Rate')
#  plt.ylabel('True Positive Rate')
#  plt.legend(loc='lower right')
#  plt.show()

# test0()
# test1()
# learningRateTest(0.01)
# learningRateTest(0.1)
# learningRateTest(0.5)
# learningRateTest(1.0)
# learningRateTest(1.5)
# outputSplitSet()
# outputSplitSet2()
# plotROCCurves()
# plotROCCurve2("a2shuffled_set1.data", 'blue')
# plotROCCurve("a2shuffled_set1.data", 'blue')
# plotROCCurve("a2shuffled_set2.data", 'blue')
# plotROCCurve("a2shuffled_set3.data", 'blue')
# plotROCCurve("a2shuffled_set4.data", 'blue')
# plotROCCurve("a2shuffled_set5.data", 'blue')
plotError("a2shuffled_set3.data")