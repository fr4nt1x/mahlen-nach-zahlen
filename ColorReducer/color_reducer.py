import numpy as np


class ColorReducer(object):
    def __init__(self, input_image, colors):
        self._input_image = input_image
        self.output_image = input_image
        self._colors = colors

    def reduce_color(self):
        new_array_shape = list(self._input_image.shape[0:2])
        new_array_shape.append(len(self._colors))

        color_array = np.zeros(shape=new_array_shape)

        for index, color in enumerate(self._colors):
            difference_color = np.linalg.norm(color - self._input_image, axis=2)
            color_array[:, :, index] = difference_color

        minimum_dist_index = np.argmin(color_array, axis=2)
        self.output_image = np.array(self._colors[minimum_dist_index], dtype=np.uint8)