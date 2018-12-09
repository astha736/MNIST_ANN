from mnist import MNIST
import numpy as np
# import math

class DataUtility:
	@staticmethod
	def loadData(absFilename,test=False):
		mndata_ = MNIST(absFilename)
		if(test == False):
			images_, labels_ = mndata_.load_training()
		else:
			images_, labels_ = mndata_.load_testing()
		return images_,labels_

	@staticmethod
	def normalizeData(images,labels):
		# normalize by float, so that all values are converted into float
		# no_of_data_points x dimension_of_data
		numpyImages_ = np.array(images)/255.0
		numpyLabels_ = np.array(labels)
		return numpyImages_, numpyLabels_

	# change sqrt function
	@staticmethod
	def getInitializedWeightMatrix(rowLength, columnLength, sigma):
		return np.random.normal(0,1/np.sqrt(sigma),(rowLength,columnLength))

	# assumption input: dim_of_integer_lable(1) x no_of_data_points(10)
	# assumption output: dim_of_binary_lable(10) x no_of_data_points(10)
	@staticmethod
	def convertLabelsToBinary(batchLabels):
		labels_ = []
		# convert labels to data point per row form
		batchLabelsTranspose = batchLabels.transpose()
		for batchLabel in batchLabels[0]:
			label_ = np.zeros(10)
			label_[batchLabel] = 1
			labels_.append(label_)
		# such that each colum is a data_label
		return np.array(labels_).transpose()

	# assumption input:dim_of_binary_lable(10) x no_of_data_points(10)
	# assumption output: dim_of_integer_lable(1) x no_of_data_points(10)
	@staticmethod
	def convertLabelsToInteger(activationLables):
		activationLablesTranspose = activationLables.transpose()
		row_,col_ = activationLablesTranspose.shape
		numericalLables_ = []
		for r in range(0,row_):
			index = np.argmax(activationLablesTranspose[r])
			numericalLables_.append(index)
		return np.array(numericalLables_).reshape(1,row_)

	@staticmethod
	def computePrediction(outputLabels,testLabels):
		correctPredict_ = 0
		outputLabels_ = outputLabels.transpose()
		testLabels_ = testLabels.transpose()
		row_, col_ = outputLabels_.shape
		for r in range(0,row_):
			# if outputLabels_[r] == testLabels_[r]:
			for c in range(0,col_):
				if outputLabels_[r][c] == testLabels_[r][c]:
					correctPredict_ = correctPredict_ + 1

		return correctPredict_
