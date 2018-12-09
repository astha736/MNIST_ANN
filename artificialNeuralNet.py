import numpy as np
from mathUtility import MathUtility
from dataUtility import DataUtility

class ArtificialNeuralNet:

	def __init__(self,weightLayerList, biasLayerList,learningRate,batchSize):
		self.weightLayerList_ = weightLayerList
		self.biasLayerList_ = biasLayerList
		self.alpha_ = learningRate
		self.batchSize_ = batchSize

		self.activationList_ = []
		self.zList_ = []
		self.errorList_ = []

	def resetInternalEvaluation(self):
		self.activationList_ = []
		self.zList_ = []
		self.errorList_ = []

	def forwardComputationPerLayer(self,activationPreviousLayer, weightCurrentLayer, biasCurrentLayer):
		oneVector = np.ones(self.batchSize_).reshape((1, self.batchSize_)) # 1 x m size

		zLayer = np.dot(weightCurrentLayer,activationPreviousLayer)+np.dot(biasCurrentLayer,oneVector)
		
		self.zList_.append(zLayer)

		# method to make a function for numpy sigmoid
		activation_ = MathUtility.matrixLogisticFunction(zLayer)

		return activation_

	def forwardComputation(self,inputData):

		tempActivationLayer = inputData # 784 x m
		# activationList_ = []

		self.activationList_.append(inputData) # noofLayers+1 x 784 x m

		for layerIndex in range(0,len(self.weightLayerList_)):
			tempActivationLayer = self.forwardComputationPerLayer(
				tempActivationLayer,
				self.weightLayerList_[layerIndex],
				self.biasLayerList_[layerIndex])
			# (nl+1) x (nl x batchSize)
			self.activationList_.append(tempActivationLayer)
		return

	def backwardComputationLayer(self,zForlayer,weightForNextLayer,errorForNextLayer):

		# method to make a function for numpy sigmoid
		logisticDerivativeThisLayer = MathUtility.matrixLogisticDerivativeFunction(zForlayer)
		var1 = np.dot(weightForNextLayer.transpose(),errorForNextLayer)
		layerError =  np.multiply(var1,logisticDerivativeThisLayer)
		return layerError


	def backwardComputation(self,batchLabelBinary):
		tempActualWeight = batchLabelBinary 
		tempErrorLayer = self.activationList_[-1] - batchLabelBinary
		self.errorList_.append(tempErrorLayer)
		for layerIndex in range(len(self.zList_)-2,-1,-1):
			tempErrorLayer = self.backwardComputationLayer(
				self.zList_[layerIndex],
				self.weightLayerList_[layerIndex+1],
				tempErrorLayer)
			self.errorList_.insert(0,tempErrorLayer)
		return



	def updateWeightsBiases(self):
		for index in range(0,len(self.weightLayerList_)):
			self.weightLayerList_[index] = self.weightLayerList_[index] - (self.alpha_/float(self.batchSize_))*np.dot(
				self.errorList_[index],
				self.activationList_[index].transpose())
			oneVector = np.ones(self.batchSize_).reshape((self.batchSize_, 1))
			self.biasLayerList_[index] = self.biasLayerList_[index] - (self.alpha_/float(self.batchSize_))*np.dot(
				self.errorList_[index],
				oneVector)
		# set these to null again for each loop new values are computed
		self.resetInternalEvaluation()
		return

	def computeForwardTest(self,inputData):
		self.forwardComputation(inputData)
		lastActivation = self.activationList_[-1]
		self.resetInternalEvaluation()
		return lastActivation