import cv2 as cv
import numpy as np
from colorthief import ColorThief
from AreaSelector.area_collector import AreaCollector

if __name__ == "__main__":
    path_to_image = 'data/chris.jpeg'

    image_array = cv.imread(path_to_image)
    image_array = cv.cvtColor(image_array, cv.COLOR_BGR2RGB)
    color_thief = ColorThief(path_to_image)

    # build a color palette

    colors = np.unique(np.asarray(color_thief.get_palette(color_count=8, quality=10), dtype=np.int64), axis=0)
    # colors = np.array([ [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255],
    #                  [255, 0, 255]])

    area_collector = AreaCollector(image_array, colors, 10000)
    area_collector.preprocess_image()
    area_image = area_collector.find_areas()
    area_collector.save_output('./data/out/out.bmp')
    area_collector.save_input('./data/out/in.bmp')
    area_collector.save_color_image('./data/out/color.bmp')
    # Image.fromarray(color_reducer.output_image).filter(ImageFilter.SHARPEN).save(path_output_image)
