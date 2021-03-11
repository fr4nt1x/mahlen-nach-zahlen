from PIL import Image
import numpy as np

def create_grey_colors():
    colors = np.multiply(np.tile(np.arange(0, 254, dtype=np.int64), 3).reshape((3,254)).transpose(), np.ones((254, 3), dtype=np.int64))
    return colors
if __name__ == "__main__":

    colors = create_grey_colors()
    im = Image.open('data/test2.jpg')

    image_array = np.array(im)
    new_image_array = np.zeros(image_array.shape, dtype=np.uint8)
    new_array_shape = list(image_array.shape[0:2])
    new_array_shape.append(len(colors))
    color_array = np.zeros(shape=new_array_shape)
    for index, color in enumerate(colors):
        difference_color = np.linalg.norm(color-image_array, axis=2)
        color_array[:, :, index] = difference_color

    minimum_dist_index = np.argmin(color_array, axis=2)
    new_image_array = np.array(colors[minimum_dist_index],dtype=np.uint8)
    new_image = Image.fromarray(new_image_array)
    new_image.save('data/testoutput.jpg')