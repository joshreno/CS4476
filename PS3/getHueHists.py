import numpy as np
import matplotlib.pyplot as plot
from skimage.color import rgb2hsv
from quantizeHSV import quantizeHSV

def getHueHists(im, k):
	plot.subplot(2,2,3)
	histEqual, _ , _ = plot.hist(np.reshape(rgb2hsv(im)[:, :, 0], (-1, 1)), bins = k)
	outputImg, meanHues = quantizeHSV(im, k)
	edges = np.zeros(((k + 1), 1))
	edges[1: -1] = ((np.sort(meanHues, axis = 0)[:-1] + np.sort(meanHues, axis = 0)[1:]) / 2)
	edges[0], edges[-1] = max(np.sort(meanHues, axis = 0)[0] - edges[1], 0), min(np.sort(meanHues, axis = 0)[-1] + edges[-2], 1)
	edges = edges[:,0]
	plot.subplot(2,2,4)
	histClustered, _ , _ = plot.hist(np.reshape(rgb2hsv(outputImg)[:, :, 0],(-1, 1)), bins = edges)
	return histEqual, histClustered