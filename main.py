import cv2 as cv
import numpy as np
from colorthief import ColorThief
from AreaSelector.area_collector import AreaCollector

if __name__ == "__main__":
    path_to_image = 'data/chris.jpeg'
    path_output_image = 'data/rectOut.bmp'

    image_array = cv.imread(path_to_image)
    image_array = cv.cvtColor(image_array, cv.COLOR_BGR2RGB)
    color_thief = ColorThief(path_to_image)

    # build a color palette

    colors = np.unique(np.asarray(color_thief.get_palette(color_count=8, quality=10), dtype=np.int64), axis=0)
    # colors = np.array([ [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255],
    #                  [255, 0, 255]])

    area_collector = AreaCollector(image_array, colors, 500)
    area_collector.preprocess_image()
    area_image = area_collector.find_areas()
    # Image.fromarray(color_reducer.output_image).filter(ImageFilter.SHARPEN).save(path_output_image)
