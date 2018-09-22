import numpy as np
import matplotlib.pyplot as plt

def partA(A):
	plt.plot(np.arange(10000) ,(np.sort(A, axis = None))[::-1])
	plt.show()

def partB(A):
	plt.hist(A, bins = 20)
	plt.show()

def partC(A):
	x = A[50:,0:50]
	plt.imshow(x, interpolation = 'none')
	plt.savefig('question4partC.png')
	np.save('outputXPS0Q1.npy', x)

def partD(A):
	y = A - A.mean()
	plt.imshow(y)
	plt.savefig('question4partD.png')
	np.save('outputYPS0Q1.npy', y)

def partE(A):
	threshold = A.mean()
 	z = np.zeros((100,100,3),'float64')
 	z[A > threshold] = [1, 0, 0]
 	np.save('outputZPS0Q1.png', z)
	plt.imshow(z)
	plt.show()

def main():
	A = np.load('inputAPS0Q1.npy')
	#partA(A)
	#partB(A)
	partC(A)
	partD(A)
	#partE(A)

if __name__ == "__main__":
    main()