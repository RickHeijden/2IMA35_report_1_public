import math
import sys

import numpy as np
from sklearn.datasets import make_circles


class DataModifier:

    def __init__(self):
        pass

    def add_gaussian_noise(self, V, noise=0.1):
        max_x, max_y = -1.0, 1.0
        min_x, min_y = sys.maxsize, sys.maxsize
        for v in V:
            max_x = max(max_x, v[0])
            max_y = max(max_y, v[1])
            min_x = min(min_x, v[0])
            min_y = min(min_y, v[1])

        size = int(np.ceil(len(V) * noise))
        for i in range(size):
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            V.append((x, y))
        return V, len(V)

    def add_clustered_noise(self, V, type='', noise=0.1):
        max_x, max_y = -1.0, 1.0
        min_x, min_y = sys.maxsize, sys.maxsize
        for v in V:
            max_x = max(max_x, v[0])
            max_y = max(max_y, v[1])
            min_x = min(min_x, v[0])
            min_y = min(min_y, v[1])

        size = int(np.ceil(len(V) * noise))
        if type == '':
            NotImplementedError('No parameter given')
        elif type == 'circle':
            center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
            radius = min(max_x - min_x, max_y - min_y) * 0.7
            pi = math.pi
            for i in range(size):
                V.append((math.cos(2 * pi / size * i) * radius + center[0],
                          math.sin(2 * pi / size * i) * radius + center[1]))
        elif type == 'horizontal_line':
            y_center = (max_y + min_y) / 2
            interval = (max_y - min_y) / size
            current_x = min_x
            for i in range(size):
                V.append((current_x, y_center))
                current_x += interval
        elif type == 'vertical_line':
            x_center = (max_x + min_x) / 2
            interval = (max_x - min_x) / size
            current_y = min_y
            for i in range(size):
                V.append((x_center, current_y))
                current_y += interval
        else:
            NotImplementedError('%s not implemented' % type)
        return V, len(V)

    def delete_data(self, V, dataset=''):
        if dataset == '':
            NotImplementedError('No parameter given')
        else:
            NotImplementedError('%s not implemented' % dataset)
