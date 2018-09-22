import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from computeH import computeH
from scipy.misc import imread,imsave


def warpImage(inputIm, refIm, H):
    nRow, nCol, _ = inputIm.shape
    n = np.dot(H, np.array([[0, nCol - 1, nCol - 1, 0],
        [0 , 0, nRow - 1, nRow - 1],
        [1, 1, 1, 1]]))
    all2 = n[2, :]
    t = np.zeros((n.shape[0] - 1, n.shape[1]))
    t[0, :] = np.around(n[0, :] / all2)
    t[1, :] = np.around(n[1, :] / all2)
    p, q = np.meshgrid(np.arange(min(np.amin(t[0,:]),0), np.amax(t[0,:]), 1),
        np.arange(min(np.amin(t[1,:]),0), np.amax(t[1,:]), 1))
    nc = np.dot(inv(H), np.array([np.ravel(p),np.ravel(q),np.ravel(np.ones(p.shape))]))
    c = np.zeros((nc.shape[0] - 1, nc.shape[1]))
    c[0,:] = np.around(nc[0,:] / nc[2,:])
    c[1,:] = np.around(nc[1,:] / nc[2,:])
    c = np.reshape(np.transpose(c),(p.shape[0],p.shape[1],2))
    warpIm = np.zeros((p.shape[0],p.shape[1],3))
    for row in range(0, warpIm.shape[0]):
        for col in range(0, warpIm.shape[1]):
            if c[row, col, 0] < 0 or c[row, col, 0] >  nCol - 1 or c[row, col, 1] < 0 or c[row, col, 1] > nRow - 1: warpIm[row, col] = np.array([0, 0, 0])
            else: warpIm[row, col] = np.array(inputIm[int(c[row, col, 1]), int(c[row, col, 0]),:])
    mergeIm = np.copy(warpIm)
    mergeIm = np.asarray(mergeIm, dtype = np.uint8)
    mergeIm[abs(int(min(np.amin(t[1,:]),0))): abs(int(min(np.amin(t[1,:]),0))) + refIm.shape[0], abs(int(min(np.amin(t[0, :]),0))):abs(int(min(np.amin(t[0, :]),0))) + refIm.shape[1], :] = refIm
    return np.asarray(warpIm, dtype=np.uint8), mergeIm

if __name__ == "__main__":
    # crops
    t1 = np.load('cc1.npy').transpose()
    t2 = np.load ('cc2.npy').transpose()
    H = computeH(t1, t2)
    warpIm, mergeIm = warpImage(imread('crop1.jpg'),imread('crop2.jpg'),computeH(t1,t2))
    plt.figure(1)
    plt.imshow(mergeIm)
    plt.figure(2)
    plt.imshow(warpIm)
    plt.show()

    # wdc
    t1 = np.array([[218, 118, 191, 95, 378, 382, 339, 302], [148, 207, 116, 179, 243, 137,  119,  95]])
    t2 = np.array([[307, 323, 364, 375, 114, 162, 224, 302], [159, 35, 145,  46, 128, 276, 269, 287]])
    points = np.empty([2, 2, 8])
    points[0] = t1
    points[1] = t2
    np.save('points.npy', points)
    warpIm, mergeIm = warpImage(imread('wdc1.jpg'),imread('wdc2.jpg'),computeH(t1,t2))
    plt.figure(3)
    plt.imshow(mergeIm)
    plt.figure(4)
    plt.imshow(warpIm)
    plt.show()

    # warpIm, mergeIm = warpImage(imread('IMG_3857.JPG'),imread('IMG_3856.JPG'),computeH(t2,t1))