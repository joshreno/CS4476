import numpy as np

def predictAction(testMoments, trainMoments, trainLabels):
    distance = np.zeros((1, trainMoments.shape[0]))
    for i in range(0, trainMoments.shape[0]):
        distance[:,i] = np.sqrt(np.sum(np.divide(np.power(np.reshape(trainMoments[i,:], (-1, 1)) - 
            np.reshape(testMoments,(-1, 1)), 2), np.reshape(np.nanvar(trainMoments, axis = 0), (-1, 1)))))
    return int(trainLabels[np.argsort(distance)][0, 0])

if __name__ == "__main__":
    print predictAction(np.load('rightkick-p1-1_huVector.npy'), 
        np.asarray(np.load('huVectors.npy')), 
        np.array([[1], [1], [1], [1], [2], [2], [2], [2], [3], [3], [3], [3], [4], [4],[ 4], [4], [5], [5], [5], [5]]))