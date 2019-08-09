from logistic_reg import *

def testDataLoad():
  print("Running " + testDataLoad.__name__)
  X = dataLoad("data_banknote_authentication.txt")
  print(X.shape)

def testNormalization():
  print("Running " + testNormalization.__name__)
  X = dataLoad("data_banknote_authentication.txt")
  X_norm = dataNorm(X)
  testNorm([X_norm])

def testErrorCompute():
  print("Running " + testErrorCompute.__name__)
  X = dataLoad("data_banknote_authentication.txt")
  X_norm = dataNorm(X)
  J = errCompute(X_norm, np.zeros((X_norm.shape[1] - 1, 1)))
  print(J)

def testSGD():
  print("Running " + testSGD.__name__)
  X = dataLoad("a4shuffled.data")
  X_shufnorm = dataNorm(X)
  theta = stochasticGD(X_shufnorm, np.zeros((X_shufnorm.shape[1] - 1, 1)), 0.01, 1372 * 20)
  J = errCompute(X_shufnorm, theta)
  print("Error : {}".format(J))

def testSGDPredict():
  print("Running " + testSGDPredict.__name__)
  print("Train set : a4shuffled.data")
  X = dataLoad("a4shuffled.data")
  X_shufnorm = dataNorm(X)
  theta = stochasticGD(X_shufnorm, np.zeros((X_shufnorm.shape[1] - 1, 1)), 0.01, 1372 * 20)
  J = errCompute(X_shufnorm, theta)
  print("Error : {}".format(J))

  print("Test set : a4shuffled.data")
  y_predict = Predict(X_shufnorm, theta)

  test_predict = []
  count = 0

  dataFile = open("a4predict.data", "r")
  lines = dataFile.readlines()

  for line in lines:
    words = line.split(",")
    for word in words:
      test_predict.append(int(word))
  
  hit = 0
  for i in range(len(test_predict)):
    if test_predict[i] == y_predict[i]:
      hit += 1
  accuracy = hit / len(test_predict) * 100.0
  print("Accuracy : {}".format(accuracy))

def testSGDPredict2():
  print("Running " + testSGDPredict2.__name__)
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

testDataLoad()
testNormalization()
testErrorCompute()
testSGD()
testSGDPredict()
testSGDPredict2()