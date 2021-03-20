import numpy as np
import cv2 as cv


class AreaCollector(object):
    def __init__(self, input_image, colors):
        self._input_image = input_image
        self._colors = colors

    def show(self, image):
        image_array = image
        if len(image.shape) >= 3:
            image_array = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        cv.imshow('area-collector', image_array)
        cv.waitKey(0)

    def find_areas(self):
        self.show(self._input_image)
        index = 0
        for color in self._colors:
            print("Processing Color (R,G,B) : {}".format(color))
            mask = np.array(np.all(self._input_image == np.asarray(color), axis=2), dtype=np.uint8)
            img_one_color = cv.bitwise_and(self._input_image, self._input_image, mask=mask)
            self.show(img_one_color)
            image_gray = cv.cvtColor(img_one_color, cv.COLOR_RGB2GRAY)
            contours, hierachy = cv.findContours(image_gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            output = 255*np.ones(self._input_image.shape, dtype=np.uint8)
            cv.drawContours(output, contours, -1, (0, 0, 0), 1)
            self.show(output)

            for contour in contours:
                rect = cv.minAreaRect(contour)
                font = cv.FONT_HERSHEY_SIMPLEX
                font_color = (0, 255, 0)
                text = str(index)
                font_size = 0.25
                font_thickness = 1
                output = cv.putText(output, text, (int(rect[0][0]), int(rect[0][1])), font, font_size, font_color,
                                    font_thickness, cv.LINE_AA)
                self.show(output)
            index += 1
