import matplotlib.pyplot as plot
from scipy.misc import imread
from quantizeRGB import quantizeRGB
from quantizeHSV import quantizeHSV
from computeQuantizationError import computeQuantizationError
from getHueHists import getHueHists
from scipy.misc import imsave

def colorQuantizeMain(input_image, k):
    output_quantizeRGB_image, _ = quantizeRGB(input_image, k)
    plot.subplot(2,2,1)
    plot.imshow(output_quantizeRGB_image)
    error_RGB = computeQuantizationError(input_image, output_quantizeRGB_image)
    output_quantizeHSV_image, _ = quantizeHSV(input_image, k)
    plot.subplot(2,2,2)
    plot.imshow(output_quantizeHSV_image)
    error_HSV = computeQuantizationError(input_image, output_quantizeHSV_image)
    hist_equal, hist_clustered = getHueHists(input_image, k)
    plot.show()
    return error_RGB, error_HSV, hist_equal, hist_clustered

# error_RGB_2,error_HSV_2,hist_equal_2,hist_clustered_2 = colorQuantizeMain(imread('fish.jpg'), 2)
# print "k = 2"
# print "RGB = " + str(error_RGB_2)
# print "HSV = " + str(error_HSV_2)
# print str(hist_equal_2)
# print str(hist_clustered_2)