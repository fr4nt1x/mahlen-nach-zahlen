import numpy as np


class AreaCollector(object):
    def __init__(self, input_image):
        self._input_image = input_image

    def find_areas(self):
        gradient = np.gradient(self._input_image, axis=(0, 1))
        gradient_x = gradient[0]
        gradient_y = gradient[1]
        x_equal = gradient_x == 0
        y_equal = gradient_y == 0
        x_equal = x_equal.all(axis=2)
        y_equal = y_equal.all(axis=2)
        both_equal = np.logical_and(x_equal, y_equal)
        output = np.ones(self._input_image.shape, dtype=np.uint8)*255
        output[np.logical_not(both_equal)] = 0

        return output