import numpy as np

def classifyAllActions(testMoments, trainMoments, trainLabels, K):
    distance = np.zeros((1, trainMoments.shape[0]))
    for i in range(0, trainMoments.shape[0]):
        nanvar = np.power(np.sqrt(np.reshape(np.nanvar(trainMoments,axis=0),(-1,1))),2)
        distance[:,i] = np.sqrt(np.sum(np.divide(np.power(np.reshape(trainMoments[i,:],(-1,1)) - 
            np.reshape(testMoments,(-1,1)),2), nanvar)))
        #Minkowski distanceo of order 3
        # distance[:,i] = np.cbrt(np.sum(np.power(np.reshape(trainMoments[i,:],(-1,1)) - 
        #     np.reshape(testMoments,(-1,1)),3)))
        #Manhattan distance
        # distance[:,i] = np.sum(np.power(np.reshape(trainMoments[i,:],(-1,1)) - 
        #     np.reshape(testMoments,(-1,1)),1))
        # non normalized
        # distance[:,i] = np.sqrt(np.sum(np.power(np.reshape(trainMoments[i,:],(-1,1)) - 
        #     np.reshape(testMoments,(-1,1)),2)))
    return np.argmax(np.bincount(trainLabels[np.argsort(distance)].flatten()[0:K]))

if __name__ == "__main__":
    trainMoments = np.asarray(np.load('huVectors.npy'))
    trainLabels = np.array([[1],[1],[1],[1],[2],[2],[2],[2],[3],[3],[3],[3],[4],[4],[4],[4],[5],[5],[5],[5]])
    K = 4
    predicted_actions = np.zeros((trainLabels.shape), dtype=np.float64)
    confusion_matrix = np.zeros((5,5), dtype=np.float64)
    mean_recognition_rate = np.zeros((5,1))
    row, _ = trainMoments.shape
    for i in range(0,row):
        predicted_actions[i,:] = classifyAllActions(trainMoments[i,:], 
            np.delete(trainMoments,i,0), 
            np.delete(trainLabels,i,0), 
            K)
    for i in range(0,row):
        confusion_matrix[int(trainLabels[i,:][0]-1), int(predicted_actions[i,:][0]-1)] = \
            confusion_matrix[
            int(trainLabels[i,:][0]-1),
            int(predicted_actions[i,:][0]-1)] + \
            1
    #np.save("confusion_matrix.npy", confusion_matrix)
    for i in range(0, confusion_matrix.shape[1]):
        mean_recognition_rate[i,:] = np.float(confusion_matrix[i,i] / 4)
    #np.save("mean_recognition_rate.npy", mean_recognition_rate)
    print (mean_recognition_rate)
    print (confusion_matrix)
