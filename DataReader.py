import json
import numpy as np


class DataReader:

    def __init__(self):
        pass

    def read_json(self, file_loc):
        with open(file_loc) as file:
            data = json.load(file)

            x = data['x']
            y = data['y']
            edge_start = data['edge_i']
            edge_end = data['edge_j']

            vertex_coordinates = []
            vertices = []
            for i in range(data['n']):
                vertices.append(i)
                vertex_coordinates.append([x[i], y[i]])

            edges = dict()
            for i in range(data['m']):
                if edge_start[i] not in edges:
                    edges[edge_start[i]] = {edge_end[i]: np.sqrt((y[edge_start[i]] - y[edge_end[i]]) ** 2 +
                                                                 (x[edge_start[i]] - x[edge_end[i]]) ** 2)}
                else:
                    edges[edge_start[i]][edge_end[i]] = np.sqrt((y[edge_start[i]] - y[edge_end[i]]) ** 2 +
                                                                (x[edge_start[i]] - x[edge_end[i]]) ** 2)

        return vertices, data['m'], edges, vertex_coordinates
