import numpy as np
	
def computeH(t1, t2):
    t1_1 = t1.shape[1]
    t1 = np.append(t1,np.ones((1, t1_1)), axis = 0)
    t1_1double = t1_1 * 2
    L = np.zeros((t1_1double, 9))
    i = 0
    for row in range(0, t1_1double, 2):
        t10i = t1[0, i]
        t11i = t1[1, i]
        t12i = t1[2, i]
        t20i = t2[0, i]
        t21i = t2[1, i]
        L[row] = [t10i, t11i, t12i, 0, 0, 0, - t20i * t10i, - t20i * t11i, - t20i * t12i]
        L[row + 1] = [0, 0, 0, t10i, t11i, t12i, - t21i * t10i, - t21i * t11i, - t21i * t12i]
        i += 1
    eigenvalues, eigenvectors = np.linalg.eig(np.dot(L.T, L))
    return np.reshape(eigenvectors[:, np.argmin(eigenvalues)],(3, 3))