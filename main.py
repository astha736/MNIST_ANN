import numpy as np
import math

from dataUtility import DataUtility
from batchUtility import BatchUtility
from mathUtility import MathUtility
from artificialNeuralNet import ArtificialNeuralNet

trainBatchSize = 8
testBatchSize = 8
mnistDataFolderName = '/home/astha/EMARO/ARTIN/Lab3/python-mnist/data'

noOfEpochs = 10

# ########################Training Data and Params ########################################
# load and normalize data

trainDataImages, trainDataLabels	= DataUtility.loadData(mnistDataFolderName)
trainNpImages, trainNpLables  		= DataUtility.normalizeData(trainDataImages,trainDataLabels)

trainBU 							= BatchUtility(trainBatchSize,trainNpImages, trainNpLables)

# # ####################### Testing Data and Params ##############################################
testDataImages, testDataLabels 		= DataUtility.loadData(mnistDataFolderName,True)
testNpImages, testNpLables  		= DataUtility.normalizeData(testDataImages,testDataLabels)

testBU 								= BatchUtility(trainBatchSize,testNpImages, testNpLables)

# #######################  General ################################################

layer1 = DataUtility.getInitializedWeightMatrix(30,784,784)
layer2 = DataUtility.getInitializedWeightMatrix(10,30,30)

weightLayerList = [layer1,layer2]

bias1 = np.zeros((30,1))
bias2 = np.zeros((10,1))

biasLayerList = [bias1,bias2]

learningRate = 0.05
ann = ArtificialNeuralNet(weightLayerList,biasLayerList,learningRate,trainBatchSize)

for epoch in range(0,noOfEpochs):
	print "Epoch:", epoch
	for batchIndex in range(0,trainBU.maxbatchIndex_): #maxbatchIndex
		batchImages, batchLabels = trainBU.getNextMiniBatch()
		batchLabelBinary = DataUtility.convertLabelsToBinary(batchLabels)
		ann.forwardComputation(batchImages)
		ann.backwardComputation(batchLabelBinary)
		ann.updateWeightsBiases()

	NoOfCorrectTest = 0
	for testBatchIndex in range(0,testBU.maxbatchIndex_): #testMaxBatchIndex
		testMBImages, testMBLabels = testBU.getNextMiniBatch()
		outputActivation = ann.computeForwardTest(testMBImages)
		temp = DataUtility.convertLabelsToInteger(outputActivation)
		NoOfCorrectTest = NoOfCorrectTest + DataUtility.computePrediction(temp,testMBLabels)

	print "NoOfCorrectTest: ",NoOfCorrectTest/ float(len(testNpImages))