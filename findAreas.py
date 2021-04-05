import numpy as np
import time


class FindAreas(object):
    def __init__(self, image_array):
        self._image_array = image_array
        self.size_x = self._image_array.shape[0]
        self.size_y = self._image_array.shape[1]
        self.total_pixels = self.size_y*self.size_x

    def find_areas(self):
        visited = np.zeros(self._image_array.shape[0:2])
        group_index_matrix = -1*np.ones((self.size_x, self.size_y), dtype=np.int64)
        current_list = []
        current_group_index = 0
        number_processed = 0
        visited_dict = {x: self._convert_from_index(x) for x in range(0, self.total_pixels)}

        while number_processed < self.total_pixels:
            #not_visited = np.asarray(visited == 0).nonzero()
            not_visited = visited_dict[list(visited_dict)[0]]
            #current_list.append((not_visited[0][0], not_visited[1][0]))
            current_list.append(not_visited)
            while len(current_list) > 0:
                current_index = current_list.pop(0)
                print("Processed, Total:")
                print(number_processed, self.size_x*self.size_y)
                number_processed += 1
                group_index_matrix[current_index] = current_group_index
                visited[current_index[0], current_index[1]] = 1
                #print("Index")
                #print(current_index[0], current_index[1])
                #print(self._convert_to_index(current_index[0], current_index[1]))
                visited_dict.pop(self._convert_to_index(current_index[0], current_index[1]))
                current_pixel = self._image_array[current_index]

                x = current_index[0]-1
                y = current_index[1]

                if 0 <= x < self.size_x and 0 <= y < self.size_y:
                    if visited[x, y] == 0 and self._image_array[x, y] == current_pixel:
                        if (x, y) not in current_list:
                            current_list.append((x, y))
                x = current_index[0]
                y = current_index[1]-1
                if 0 <= x < self.size_x and 0 <= y < self.size_y:
                    if visited[x, y] == 0 and self._image_array[x, y] == current_pixel:
                        if (x, y) not in current_list:
                            current_list.append((x, y))
                x = current_index[0]+1
                y = current_index[1]
                if 0 <= x < self.size_x and 0 <= y < self.size_y:
                    if visited[x, y] == 0 and self._image_array[x, y] == current_pixel:
                        if (x, y) not in current_list:
                            current_list.append((x, y))
                x = current_index[0]
                y = current_index[1]+1
                if 0 <= x < self.size_x and 0 <= y < self.size_y:
                    if visited[x, y] == 0 and self._image_array[x, y] == current_pixel:
                        if (x, y) not in current_list:
                            current_list.append((x, y))
            current_group_index += 1
        return group_index_matrix

    def _convert_to_index(self, x, y):
        if x < 0 or x >= self.size_x:
            raise Exception("X out of bounds")
        if y < 0 or y >= self.size_y:
            raise Exception("Y out of bounds")
        return int(self.size_y * x + y)

    def _convert_from_index(self, i):
        if i > self.total_pixels:
            raise Exception("Index to big")
        rest = i % self.size_y
        y = rest
        x = int(i/self.size_y)
        return int(x), int(y)


if __name__=="__main__":
    times = []
    for i in range(9, 10):
        image = np.random.randint(0, 4, size=(i*100, i*100))
        #image = np.zeros((10, 10))
#         image = np.array([[2,2,0,1,1],
# [1,0,2,2,3],
# [0,0,1,2,1],
# [0,0,3,1,1],
# [2,3,3,3,2]])
        print(image)
        fa = FindAreas(image)
        startTime = time.time()
        fa.find_areas()
        executionTime = (time.time() - startTime)
        times.append(executionTime)

    print('Execution time in seconds: ' + str(times))
