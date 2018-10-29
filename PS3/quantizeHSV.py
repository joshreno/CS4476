import numpy as np
from skimage.color import rgb2hsv, hsv2rgb
from scipy.cluster.vq import kmeans2

def quantizeHSV(origImg, k):
	input_HSV_image = rgb2hsv(origImg)
	meanHues, labels = kmeans2(np.reshape(input_HSV_image[:, :, 0], (-1, 1)), k)
	output_HSV_image = input_HSV_image
	output_HSV_image[:, :, 0] = np.reshape(meanHues[labels], (input_HSV_image[:, :, 0].shape))
	return (np.around(hsv2rgb(output_HSV_image) * 255)).astype(np.uint8), meanHues