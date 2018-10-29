import numpy as np
from scipy.cluster.vq import kmeans2

def quantizeRGB(origImg, k):
	input_image = origImg.astype(np.float64)
	meanColors, labels = kmeans2(np.reshape(input_image, (-1, 3)), k)
	return np.reshape(meanColors[labels,:], (input_image.shape)).astype(np.uint8), meanColors