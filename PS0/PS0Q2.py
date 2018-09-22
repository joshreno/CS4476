import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def part1(img):
	img_swap = np.empty_like(img)
	img_swap[...,1] = img[...,0]
	img_swap[...,0] = img[...,1]
	img_swap[...,2] = img[...,2]
	plt.subplot(3, 2, 1)
	plt.imshow(img_swap)
	plt.title('G and R Channels Swapped')

def part2(img):
	img_gray = rgb2gray(img)
	plt.subplot(3, 2, 2)
	plt.imshow(img_gray, cmap = plt.get_cmap('gray'))
	#plt.imshow(img_gray, interpolation=None)
	plt.colorbar()
	plt.title('Grayscale')
	plt.show()

def part3(img):
	img_gray = rgb2gray(img)
	img_negative = 1 - img_gray
	plt.subplot(3, 2, 3)
	plt.imshow(img_negative, cmap = plt.get_cmap('gray'))
	plt.title('Negative')

def part4(img):
	img_gray = rgb2gray(img)
	img_flip = np.fliplr(img_gray)
	plt.subplot(3, 2, 4)
	plt.imshow(img_flip, cmap = plt.get_cmap('gray'))
	plt.title('Flipped')

def part5(img):
	img_gray = rgb2gray(img)
	img_flip = np.fliplr(img_gray)
	img_average = (img_gray + img_flip)/2
	plt.subplot(3, 2, 5)
	plt.imshow(img_average, cmap = plt.get_cmap('gray'))
	plt.title('Average of Grayscale and Flipped')

def part6(img):
	img_gray = rgb2gray(img)
	N = np.empty_like(img_gray)
	N = np.random.random_integers(low = 0, high = 255, size = img_gray.shape)
	np.save('noise.npy', N)
	img_gray += N
	for i in np.nditer(img_gray):
		if i > 255: i = 255
	plt.subplot(3, 2, 6)
	plt.imshow(img_gray, cmap = plt.get_cmap('gray'))
	plt.title('Noise')
	plt.show()

def rgb2gray(rgb):
	#borrowed from MATLAB rbg2gray
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def main():
	img = mpimg.imread('inputPS0Q2.png')
	part1(img)
	part2(img)
	part3(img)
	part4(img)
	part5(img)
	part6(img)

if __name__ == "__main__":
    main()