import numpy as np
import cv2 as cv


class AreaCollector(object):
    def __init__(self, input_image):
        self._input_image = cv.imread(input_image)

    def find_areas(self):

        imgray = cv.cvtColor(self._input_image, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(imgray, 127, 255, 0)
        contours, hierachy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        output = np.zeros(self._input_image.shape)
        cv.drawContours(output, contours, -1, (0, 255, 0), 1)

        cv.imshow('bla', output)
        cv.waitKey(0)