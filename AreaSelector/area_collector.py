import numpy as np
import cv2 as cv

from ColorReducer.color_reducer import ColorReducer


class AreaCollector(object):
    def __init__(self, input_image, colors, min_area=1):
        self._input_image = input_image
        self._colors = colors
        self._min_area = min_area
        self._output_image = self.get_white_image()
        self._color_image = self._create_color_image()
        self._all_contours = []

    def _create_color_image(self):
        length = 64
        font_size = 0.4
        font_thickness = 1
        font = cv.FONT_HERSHEY_SIMPLEX
        font_color = (0, 0, 0)
        image = 255 * np.ones([length + 2, length * len(self._colors), 3], dtype=np.uint8)
        index = 0
        point1 = np.array([0, 0], dtype=np.uint)
        point2 = np.array([1, 1], dtype=np.uint) * length
        midpoint = np.array([1, 1], dtype=np.uint) * np.uint((length / 2))
        for color in self._colors:
            image = cv.rectangle(image, tuple(point1.tolist()), tuple(point2.tolist()), color.tolist(), -1)

            text = str(index)
            image = cv.putText(image, text, tuple(midpoint.tolist()), font, font_size, font_color, font_thickness,
                               cv.LINE_AA)
            point1[0] += length
            point2[0] += length
            midpoint[0] += length
            index += 1
        return image

    def save_input(self, path):
        output = cv.cvtColor(self._input_image, cv.COLOR_RGB2BGR)
        cv.imwrite(path, output)

    def save_output(self, path):
        output = cv.cvtColor(self._output_image, cv.COLOR_RGB2BGR)
        cv.imwrite(path, output)

    def save_color_image(self, path):
        output = cv.cvtColor(self._color_image, cv.COLOR_RGB2BGR)
        cv.imwrite(path, output)

    def get_white_image(self):
        return 255 * np.ones(self._input_image.shape, dtype=np.uint8)

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
        output_image = cv.medianBlur(output_image, 11)
        color_reducer = ColorReducer(output_image, self._colors)
        color_reducer.reduce_color()
        output_image = color_reducer.output_image
        self.show(output_image)
        self._input_image = output_image

        for color in self._colors:
            print("Processing Color (R,G,B) : {}".format(color))
            mask = np.array(np.all(self._input_image == np.asarray(color), axis=2), dtype=np.uint8)
            img_one_color = cv.bitwise_and(self._input_image, self._input_image, mask=mask)
            image_gray = cv.cvtColor(img_one_color, cv.COLOR_RGB2GRAY)
            contours, hierarchy = cv.findContours(image_gray, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            reduced_contours = [c for c in contours if abs(cv.contourArea(c, True)) < self._min_area]
            #  reduced_contours = contours
            cv.drawContours(self._output_image, contours, -1, color.tolist(), thickness=-1, lineType=cv.LINE_8)
            cv.drawContours(self._output_image, reduced_contours, -1, [255, 255, 255], thickness=-1, lineType=cv.LINE_8)

        # self.show(self._output_image)
        # Fill white areas resulting from areas not drawn
        # some areas are total white
        color = (255, 255, 255)
        mask = np.array(np.all(self._output_image == np.asarray(color), axis=2), dtype=np.uint8)
        img_one_color = cv.bitwise_and(self._output_image, self._output_image, mask=mask)
        image_gray = cv.cvtColor(img_one_color, cv.COLOR_RGB2GRAY)

        contours, hierarchy = cv.findContours(image_gray, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
            color = self._get_neigbouring_color_for_contour(contour)
            cv.drawContours(self._output_image, contours, i, color, thickness=-1, lineType=cv.LINE_8)
        self.show(self._output_image)
        self._input_image = self._output_image
        self._output_image = self.get_white_image()

    def find_areas(self):
        index = 0
        black = np.array([0, 0, 0])
        all_contours = []
        for color in self._colors:
            print("Processing Color (R,G,B) : {}".format(color))
            mask = np.array(np.all(self._input_image == np.asarray(color), axis=2), dtype=np.uint8)
            img_one_color = cv.bitwise_and(self._input_image, self._input_image, mask=mask)
            image_gray = cv.cvtColor(img_one_color, cv.COLOR_RGB2GRAY)
            contours, hierarchy = cv.findContours(image_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            # reduced_contours = [c for c in contours if abs(cv.contourArea(c, True)) < self._min_area]
            self._all_contours += contours
            #  reduced_contours = contours
            cv.drawContours(self._output_image, contours, -1, black.tolist(), thickness=1, lineType=cv.LINE_8)
            #  cv.drawContours(self._output_image, reduced_contours, -1, [255, 255, 255], thickness=-1, lineType=cv.LINE_8)
            for contour in contours:
                self._put_color_at_contour(contour, index, black)
            # self.show(self._output_image)
            index += 1
        self.show(self._output_image)

    def _get_neigbouring_color_for_contour(self, contour):
        # TODO if white is found use alternative approach
        color = [255, 255, 255]
        # self.show(self._output_image)
        found_colors = []
        for i in range(0, contour.shape[0]):
            current_point = np.flipud(contour[i, 0, :])
            neighbor = np.array(current_point, copy=True)
            neighbor[0] -= 1
            if self.is_in_image_range(neighbor):
                current_color = self._output_image[neighbor[0], neighbor[1], :].tolist()
                if current_color != [255, 255, 255]:
                    found_colors.append(current_color)
            neighbor = np.array(current_point, copy=True)
            neighbor[0] += 1
            if self.is_in_image_range(neighbor):
                current_color = self._output_image[neighbor[0], neighbor[1], :].tolist()
                if current_color != [255, 255, 255]:
                    found_colors.append(current_color)
            neighbor = np.array(current_point, copy=True)
            neighbor[1] -= 1
            if self.is_in_image_range(neighbor):
                current_color = self._output_image[neighbor[0], neighbor[1], :].tolist()
                if current_color != [255, 255, 255]:
                    found_colors.append(current_color)
            neighbor = np.array(current_point, copy=True)
            neighbor[1] += 1
            if self.is_in_image_range(neighbor):
                current_color = self._output_image[neighbor[0], neighbor[1], :].tolist()
                if current_color != [255, 255, 255]:
                    found_colors.append(current_color)

        if found_colors:
            colors_tuple = [tuple(c) for c in found_colors]
            color = max(set(colors_tuple), key=colors_tuple.count)
        if color == [255, 255, 255]:
            print("Warning: White found should not happen.")
        return color

    def _put_color_at_contour(self, contour, index, color):
        font = cv.FONT_HERSHEY_SIMPLEX
        font_color = color.tolist()
        distance_color = 20
        text = str(index)
        font_size = 1
        font_thickness = 2
        line_thickness = 1
        draw_color = True

        if contour.shape[0] == 1:
            start_point = contour[0, 0, :]
            end_point = contour[0, 0, :]
            font_color = [255, 0, 0]
            draw_color = False
        else:
            vec = contour[0, 0, :] - contour[1, 0, :]
            mat = np.array([[0, -1], [1, 0]])
            vec_orth = np.dot(vec, mat) / np.linalg.norm(vec)
            start_point = contour[0, 0, :]
            end_point = start_point - vec_orth * distance_color
            if not self.is_in_image_range(list(reversed(end_point))):
                end_point = start_point + vec_orth * distance_color

        end_point = end_point.astype(np.int64)
        if draw_color:
            self._output_image = cv.line(self._output_image, tuple(start_point), tuple(end_point), font_color,
                                         line_thickness)
            self._output_image = cv.putText(self._output_image, text, (int(end_point[0]), int(end_point[1])), font,
                                            font_size, font_color, font_thickness, cv.LINE_8)
        else:
            # Delete previously drawn contour
            cv.drawContours(self._output_image, [contour], -1, [255, 255, 255], thickness=1, lineType=cv.LINE_8)

    def is_in_image_range(self, point):
        result = True
        shape = self._input_image.shape
        if point[0] < 0 or point[1] < 0:
            result = False
        elif point[0] >= shape[0] or point[1] >= shape[1]:
            result = False
        return result

    def save_svg_with_all_contours(self, path):
        with open(path, "w+") as f:
            f.write(
                '<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">'.format(h=self._input_image.shape[0],
                                                                                           w=self._input_image.shape[
                                                                                               1]))

            for c in self._all_contours:
                f.write('<path d="M')

                for i in range(len(c)):
                    if i == 0:
                        x0, y0 = c[i][0]
                    x, y = c[i][0]
                    f.write(f"{x} {y} ")
                f.write(f"{x0} {y0} ")
                f.write('" fill-opacity="0" style="stroke:black"/>')

            f.write("</svg>")
