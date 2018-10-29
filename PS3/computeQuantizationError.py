import numpy as np

def computeQuantizationError(origImg, quantizedImg):
	return np.sum(np.power((origImg - quantizedImg), 2)) / (origImg.shape[0] * origImg.shape[1] * origImg.shape[2])