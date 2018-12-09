import numpy as np
import math


class MathUtility:
	@staticmethod
	def logisticActivationFunction(x):
		return 1 / (1 + math.exp(-x))

	@staticmethod
	def logisticDerivativeFunction(x):
		return MathUtility.logisticActivationFunction(x)*(1-MathUtility.logisticActivationFunction(x))

	@staticmethod
	def matrixLogisticFunction(dataMatrix):
		logisticNpFunc = np.vectorize(MathUtility.logisticActivationFunction) # method to make a function for numpy sigmoid
		return logisticNpFunc(dataMatrix)

	@staticmethod
	def matrixLogisticDerivativeFunction(dataMatrix):
		logisticNpDerv = np.vectorize(MathUtility.logisticDerivativeFunction) # method to make a function for numpy sigmoid
		return logisticNpDerv(dataMatrix)
