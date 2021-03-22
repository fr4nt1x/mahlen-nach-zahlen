from PIL import Image, ImageFilter
import cv2 as cv
import numpy as np
from colorthief import ColorThief
from ColorReducer.color_reducer import ColorReducer
from AreaSelector.area_collector import AreaCollector


if __name__ == "__main__":
    path_to_image = 'data/test.jpg'
    path_output_image = 'data/reducedElefantOut.bmp'

    color_thief = ColorThief(path_to_image)
    # build a color palette
    colors = np.unique(np.asarray(color_thief.get_palette(color_count=16, quality=1), dtype=np.int64), axis=0)

    image_array = cv.imread(path_to_image)
    image_array = cv.cvtColor(image_array, cv.COLOR_BGR2RGB)

    color_reducer = ColorReducer(image_array, colors)
    color_reducer.reduce_color()

    area_collector = AreaCollector(color_reducer.output_image, colors, 2000)
    area_image = area_collector.find_areas()
    new_image_array = np.array(color_reducer.output_image)
    #Image.fromarray(color_reducer.output_image).filter(ImageFilter.SHARPEN).save(path_output_image)