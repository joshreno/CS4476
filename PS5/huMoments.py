import numpy as np

def u(p, q, H, X, Y, x_bar, y_bar):
    return np.sum(np.power((X-x_bar),p) * np.power((Y-y_bar),q) * H, dtype=np.float64)

def huMoments(H):
    np.asarray(H, dtype=np.float64)
    row, column = H.shape
    x, y = np.arange(0, column, 1), np.arange(0,row,1)
    X, Y = np.meshgrid(x, y, sparse=False, indexing='xy')
    X += 1
    np.asarray(X,dtype=np.float64)
    Y += 1
    np.asarray(X,dtype=np.float64)
    m00 = np.sum(H, dtype=np.float64)
    x_bar = np.sum(X*H, dtype=np.float64) / m00
    y_bar = np.sum(Y*H, dtype=np.float64) / m00
    u02 = u(0,2,H,X,Y,x_bar,y_bar)
    u03 = u(0,3,H,X,Y,x_bar,y_bar)
    u11 = u(1,1,H,X,Y,x_bar,y_bar)
    u12 = u(1,2,H,X,Y,x_bar,y_bar)
    u20 = u(2,0,H,X,Y,x_bar,y_bar)
    u21 = u(2,1,H,X,Y,x_bar,y_bar)
    u30 = u(3,0,H,X,Y,x_bar,y_bar)
    moments = [u20 + u02, np.power((u20 - u02),2) + 4 * np.power(u11,2), 
    np.power((u30 - 3 * u12),2) + np.power((3 * u21 - u03), 2), np.power((u30 + u12),2) + np.power((u21 + u03), 2), 
    (u30 - 3 * u12) * (u30 + u12) * (np.power((u30 + u12),2) - 3 * np.power((u21 + u03), 2)) + (3 * u21 - u03) * (u21 + u03) * (3 * np.power((u30 + u12), 2) - np.power((u21 + u03), 2)), 
    (u20 - u02) * (np.power((u30 + u12),2) - np.power((u21 + u03),2)) + 4 * u11 * (u30 + u12) * (u21 + u03), 
    (3 * u21 - u03) * (u30 + u12) * (np.power((u30 + u12),2) - 3 * np.power((u21 + u03),2)) - (u30 - 3 * u12) * (u21 + u03) * (3 * np.power((u30 + u12), 2) - np.power((u21 + u03), 2))]
    return moments

if __name__ == "__main__":
    mhi = np.load('botharms-up-p1-1_MHI.npy')
    huVector = huMoments(mhi)
    np.save('botharms-up-p1-1_huVector.npy', huVector)

    mhi = np.load('crouch-p1-1_MHI.npy')
    huVector = huMoments(mhi)
    np.save('crouch-p1-1_huVector.npy', huVector)

    mhi = np.load('rightkick-p1-1_MHI.npy')
    huVector = huMoments(mhi)
    np.save('rightkick-p1-1_huVector.npy', huVector)

    mhis = np.load('allMHIs.npy')
    row, column, num = mhis.shape
    huVectors = np.zeros((num,7))
    for i in range(num): huVectors[i,:] = huMoments(mhis[:,:,i])
    np.save('huVectors.npy', huVectors)