from PIL import Image, ImageFilter

import numpy as np
from colorthief import ColorThief
from ColorReducer.color_reducer import ColorReducer
from AreaSelector.area_collector import AreaCollector


if __name__ == "__main__":
    path_to_image = 'data/reducedElefant.bmp'
    path_output_image = 'data/reducedElefantOut.bmp'

    color_thief = ColorThief(path_to_image)
    # build a color palette
    colors = np.asarray(color_thief.get_palette(color_count=12, quality=1), dtype=np.int64)

    im = Image.open(path_to_image)
    im = im.filter(ImageFilter.GaussianBlur(2))
    #im.show()
    image_array = np.array(im)
    color_reducer = ColorReducer(image_array, colors)
    color_reducer.reduce_color()
    area_collector = AreaCollector(path_to_image)
    area_image = area_collector.find_areas()
    new_image_array = np.array(color_reducer.output_image)
    Image.fromarray(color_reducer.output_image).save(path_output_image)