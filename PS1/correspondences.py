import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import numpy as np

def getting_correspondences(i, r):
    inp = (plt.figure(1)).add_subplot(1, 1, 1)
    inp.imshow(i)
    inp = Cursor(inp, useblit=True, color='red', linewidth=1)
    ref = (plt.figure(2)).add_subplot(1, 1, 1)
    ref.imshow(r)
    ref = Cursor(ref, useblit=True, color='blue', linewidth=1)
    i = 0
    input_points = np.empty((2,0))
    ref_points = np.empty((2,0))
    try:
        while True:
            plt.figure(1)
            input_point = plt.ginput(n = 1, timeout = 0)
            input0 = input_point[0][0]
            input1 = input_point[0][1]
            plt.annotate(str(i),(input0, input1))
            plt.plot(input0, input1, 'bo')
            input_points = np.hstack((input_points,np.array([[input0], [input1]])))
            plt.figure(2)
            ref_point = plt.ginput(n = 1, timeout = 0)
            ref0 = ref_point[0][0]
            ref1 = ref_point[0][1]
            plt.annotate(str(i),(ref0, ref1))
            plt.plot(ref0, ref_point[0][1], 'go')
            ref_points = np.hstack((ref_points,np.array([[ref0],[ref1]])))
            i += 1
    except KeyboardInterrupt:
        return input_points, ref_points

if __name__ == "__main__":
    from scipy.misc import imread
    print getting_correspondences(imread('wdc1.jpg'), imread('wdc2.jpg'))