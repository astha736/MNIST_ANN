import numpy as np
# import math

class BatchUtility:
	def __init__(self,batchSize,images,labels):
		self.batchSize_ = batchSize
		self.dataImages_ = images
		self.dataLabels_ = labels
		self.noOfDataPoints_ = len(self.dataImages_)
		self.indexShuffle_ = self.getShuffleIndex(self.noOfDataPoints_)
		self.maxbatchIndex_ = self.noOfDataPoints_/self.batchSize_
		self.batchNumber_ = -1

	def getShuffleIndex(self,shuffleSize):
		# get random shuffled index
		indexArray = np.arange(shuffleSize)
		np.random.shuffle(indexArray)
		return indexArray

	# output batchImage_ = 784 x 10 (batchSize_)
	# output batchLabel_ = 1 x 10 (batchSize_)
	def getMiniBatchNumber(self,batchNumber):
		if(batchNumber >= self.maxbatchIndex_ or batchNumber < 0):
			return Null,Null
		# Use shuffled index 
		batchImage_ = np.array([self.dataImages_[self.indexShuffle_[index]] for index in range(batchNumber,batchNumber+self.batchSize_)])
		batchLabel_ = np.array([self.dataLabels_[self.indexShuffle_[index]] for index in range(batchNumber,batchNumber+self.batchSize_)])

		# col = data_points (batchSize_) and row = dimension of data
		return batchImage_.transpose(),batchLabel_.reshape((1,self.batchSize_))

	def getNextMiniBatch(self):
		if self.batchNumber_ == self.maxbatchIndex_-1:
			self.batchNumber_ = 0
		else:
			self.batchNumber_ = self.batchNumber_ + 1
		return self.getMiniBatchNumber(self.batchNumber_)
