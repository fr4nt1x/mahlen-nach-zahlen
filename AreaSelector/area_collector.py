import numpy as np
import cv2 as cv

from ColorReducer.color_reducer import ColorReducer


class AreaCollector(object):
    def __init__(self, input_image, colors, min_area=0):
        self._input_image = input_image
        self._colors = colors
        self._min_area = min_area
        self._output_image = 255 * np.ones(self._input_image.shape, dtype=np.uint8)

    def show(self, image):
        image_array = image
        if len(image.shape) >= 3:
            image_array = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        cv.imshow('area-collector', image_array)
        cv.waitKey(0)

    def preprocess_image(self):

        color_reducer = ColorReducer(self._input_image, self._colors)
        color_reducer.reduce_color()
        output_image = color_reducer.output_image
        output_image = cv.medianBlur(output_image, 31)
        color_reducer = ColorReducer(output_image, self._colors)
        color_reducer.reduce_color()
        output_image = color_reducer.output_image
        self.show(output_image)
        self._input_image = output_image

    def find_areas(self):
        index = 0

        for color in self._colors:
            print("Processing Color (R,G,B) : {}".format(color))
            mask = np.array(np.all(self._input_image == np.asarray(color), axis=2), dtype=np.uint8)
            img_one_color = cv.bitwise_and(self._input_image, self._input_image, mask=mask)
            image_gray = cv.cvtColor(img_one_color, cv.COLOR_RGB2GRAY)
            contours, hierarchy = cv.findContours(image_gray, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            reduced_contours = [c for c in contours if abs(cv.contourArea(c, True)) > self._min_area]
            #  reduced_contours = contours
            cv.drawContours(self._output_image, reduced_contours, -1, color.tolist(), thickness=-1, lineType=cv.LINE_8)

            for contour in reduced_contours:
                self._put_color_at_contour(contour, index, color)
            # self.show(self._output_image)
            index += 1
        self.show(self._output_image)

    def _put_color_at_contour(self, contour, index, color):
        font = cv.FONT_HERSHEY_SIMPLEX
        font_color = color.tolist()
        text = str(index)
        font_size = 0.4
        font_thickness = 1
        vec = contour[0, 0, :] - contour[1, 0, :]
        mat = np.array([[0, -1], [1, 0]])
        vec_orth = np.dot(vec, mat)/np.linalg.norm(vec)
        start_point = contour[0, 0, :]
        end_point = contour[0, 0, :] - vec_orth*20
        end_point = end_point.astype(np.int64)
        self._output_image = cv.line(self._output_image, tuple(start_point), tuple(end_point), font_color,
                                     font_thickness)
        self._output_image = cv.putText(self._output_image, text, (int(end_point[0]), int(end_point[1])), font,
                                        font_size, font_color, font_thickness, cv.LINE_AA)