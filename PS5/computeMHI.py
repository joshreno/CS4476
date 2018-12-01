import os
import glob
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plot

def computeMHI(directoryName):
    directory = np.sort(glob.glob(directoryName + '/' '*.pgm'))
    tau = len(directory)
    mhi = np.zeros((imread(directory[0]).shape[0], imread(directory[0]).shape[1]))
    for i in range(tau):
        depth = imread(directory[i])
        depth[depth > 39000] = 0
        depth[depth != 0] = 1
        if i > 0:
            diff = np.absolute(depth - previous)
            mhi[diff == 1] = tau
            mhi[diff != 1] = mhi[diff != 1] - 1
            mhi[mhi < 0] = 0
            previous = depth
        else:
            previous = depth
    mhi = mhi/np.amax(mhi)
    return mhi

if __name__ == "__main__":

    #Save three images
    mhi = computeMHI('./PS5_Data/botharms/botharms-up-p1-1')
    figure = plot.figure(frameon=False)
    axis = figure.add_subplot(1,1,1)
    axis.imshow(mhi)
    axis.set_title('Both Arms')
    figure.add_axes(axis)
    figure.savefig('botharms-up-p1-1_MHI.png')
    np.save('botharms-up-p1-1_MHI.npy', mhi)

    mhi = computeMHI('./PS5_Data/crouch/crouch-p1-1')
    figure = plot.figure(frameon=False)
    axis = figure.add_subplot(1,1,1)
    axis.imshow(mhi)
    axis.set_title('Crouch')
    figure.add_axes(axis)
    figure.savefig('crouch-p1-1_MHI.png')
    np.save('crouch-p1-1_MHI.npy', mhi)

    mhi = computeMHI('./PS5_Data/rightkick/rightkick-p1-1')
    figure = plot.figure(frameon=False)
    axis = figure.add_subplot(1,1,1)
    axis.imshow(mhi)
    axis.set_title('Right Kick')
    figure.add_axes(axis)
    figure.savefig('rightkick-p1-1_MHI.png')
    np.save('rightkick-p1-1_MHI.npy', mhi)

    base_directory = './PS5_Data/'
    actions = ['botharms', 'crouch', 'leftarmup', 'punch', 'rightkick']
    directoryNames = []
    mhi = np.zeros((480, 640, 20))
    for action in actions:
        directory_name = base_directory + action + '/'
        directoryNames = directoryNames + [directory_name + subdirectory_name for subdirectory_name in os.listdir(directory_name)]

    print("length: " + str(len(directoryNames)))
    for ith_directory in range(len(directoryNames)):
        temp = directoryNames[ith_directory]
        print("name: " + str(ith_directory) + " " + temp)
        if ".DS_Store" in temp:
            del directoryNames[ith_directory]
            continue
        temp = computeMHI(temp)
        mhi[:,:,ith_directory] = temp
    np.save('allMHIs.npy',mhi)