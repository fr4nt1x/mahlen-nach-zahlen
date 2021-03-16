from PIL import Image, ImageFilter

import numpy as np
from colorthief import ColorThief
from ColorReducer.color_reducer import ColorReducer
from AreaSelector.area_collector import AreaCollector


if __name__ == "__main__":
    path_to_image = 'data/Elefant.jpg'
    path_output_image = 'data/rect.bmp'

    color_thief = ColorThief(path_to_image)
    # build a color palette
    colors = np.asarray(color_thief.get_palette(color_count=12, quality=1), dtype=np.int64)

    im = Image.open(path_to_image)
    im = im.filter(ImageFilter.GaussianBlur(2))
    #im.show()
    image_array = np.array(im)
    color_reducer = ColorReducer(image_array, colors)
    color_reducer.reduce_color()
    area_collector = AreaCollector(color_reducer.output_image)
    area_image = area_collector.find_areas()
    new_image_array = np.array(color_reducer.output_image)
    Image.fromarray(color_reducer.output_image).show()
    index = area_image == 0
    index = index.all(axis=2)
    new_image_array[index, :] = 0
    Image.fromarray(new_image_array).show()



    #new_image.save(path_output_image)