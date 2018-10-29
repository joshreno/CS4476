import skimage
import math
import numpy as np
import matplotlib.pyplot as plot
from scipy.misc import imread
from scipy.misc import imsave
from scipy.ndimage.filters import convolve
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage import color
from skimage.draw import circle_perimeter

def detectCircles(im, radius, useGradient):
	edges = canny(skimage.img_as_float(rgb2gray(im)), sigma = 6)
	image = skimage.img_as_float(color.rgb2gray(im))
	gradient_x = convolve(image, np.array([[1,-1]]), mode="wrap")
	gradient_x[gradient_x == 0] = .0000000001
	gradient_direction = np.arctan((convolve(image, np.array([[1],[-1]]),mode="wrap"))/gradient_x)
	edgeRow, edgeColumn = edges.shape
	#i = edgeRow / 2
	#j = edgeColumn / 2
	#hough_space = np.zeros((i, j))
	hough_space = np.zeros((edgeRow, edgeColumn))
	for row in range(0, edgeRow):
		for column in range(0, edgeColumn):
			if edges[row, column] == 1:
				if useGradient == 0:
					for theta in np.arange(0, 2 * math.pi, .01):
						a = int(round(column + radius * math.cos(theta)))
						b = int(round(row + radius * math.sin(theta)))
						if a >= 0 and b >= 0 and a < (edgeColumn - 1) and b < (edgeRow - 1): hough_space[b, a] = hough_space[b, a] + 1
						#hough_space[b / 2, a / 2] = hough_space[b / 2, a / 2] + 1
				else:
						theta = gradient_direction[row, column]
						a = int(round(column + radius * math.cos(theta)))
						b = int(round(row + radius * math.sin(theta)))
						if a >= 0 and b >= 0 and a < (edgeColumn - 1) and b < (edgeRow - 1): hough_space[b, a] = hough_space[b, a] + 1
						#hough_space[b / 2, a / 2] = hough_space[b / 2, a / 2] + 1
						theta = gradient_direction[row, column] - math.pi
						a = int(round(column + radius * math.cos(theta)))
						b = int(round(row + radius * math.sin(theta)))
						if a >= 0 and b >= 0 and a < (edgeColumn - 1) and b < (edgeRow - 1): hough_space[b, a] = hough_space[b, a] + 1
						#hough_space[b / 2, a / 2] = hough_space[b / 2, a / 2] + 1
	plot.imshow(hough_space)
	plot.show()
	return np.column_stack(np.where(hough_space >= 0.8 * np.amax(hough_space)))

image = imread('jupiter.jpg')
centers = detectCircles(image,110,0)
for ith_circle in xrange(0,len(centers[:,0])): image[circle_perimeter(centers[ith_circle,0],centers[ith_circle,1],radius = 110, shape = image.shape)] = np.array([255,0,0])
plot.imshow(image)
plot.show()
