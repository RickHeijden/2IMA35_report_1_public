import math
from argparse import ArgumentParser
from datetime import datetime
from typing import Tuple

import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.spatial
import sklearn
import csv

from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

from pyspark import RDD, SparkConf, SparkContext
# Snap stanford

def get_clustering_data():
    """
    Retrieves all toy datasets from sklearn
    :return: circles, moons, blobs datasets.
    """
    n_samples = 1500
    noisy_circles = make_circles(n_samples=n_samples, factor=.5,
                                          noise=.05)
    noisy_moons = make_moons(n_samples=n_samples, noise=.05)
    blobs = make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None

    # Anisotropicly distributed data
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = make_blobs(n_samples=n_samples,
                                 cluster_std=[1.0, 2.5, 0.5],
                                 random_state=random_state)

    plt.figure(figsize=(9 * 2 + 3, 13))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.95, wspace=.05,
                        hspace=.01)

    plot_num = 1

    default_base = {'quantile': .3,
                    'eps': .3,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': 3,
                    'min_samples': 20,
                    'xi': 0.05,
                    'min_cluster_size': 0.1}

    datasets = [
        (noisy_circles, {'damping': .77, 'preference': -240,
                         'quantile': .2, 'n_clusters': 2,
                         'min_samples': 20, 'xi': 0.25}),
        (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
        (varied, {'eps': .18, 'n_neighbors': 2,
                  'min_samples': 5, 'xi': 0.035, 'min_cluster_size': .2}),
        (aniso, {'eps': .15, 'n_neighbors': 2,
                 'min_samples': 20, 'xi': 0.1, 'min_cluster_size': .2}),
        (blobs, {}),
        (no_structure, {})]

    return datasets


def create_distance_matrix(dataset):
    """
    Creates the distance matrix for a dataset with only vertices. Also adds the edges to a dict.
    :param dataset: dataset without edges
    :return: distance matrix, a dict of all edges and the total number of edges
    """
    x = []
    y = []
    vertices = []
    size = 0
    for line in dataset:
        vertices.append([line[0], line[1]])
    d_matrix = scipy.spatial.distance_matrix(vertices, vertices, threshold=1000000)
    dict = {}
    for i in range(len(d_matrix)):
        dict2 = {}
        for j in range(len(d_matrix[i])):
            if i != j:
                size += 1
                dict2[j] = d_matrix[i][j]
        dict[i] = dict2
    return d_matrix, dict, size, vertices


def partion_vertices(vertices, k):
    """
    Partitioning of the vertices in k smaller subsets (creates a partitioning twice
    :param vertices: all vertices
    :param k: number of subsets that need to be created
    :return: the partitioning in list format
    """
    U = []
    V = []
    random.shuffle(vertices)
    verticesU = vertices.copy()
    random.shuffle(vertices)
    verticesV = vertices.copy()
    for i in range(len(vertices)):
        if i < k:
            U.append({verticesU[i]})
            V.append({verticesV[i]})
        else:
            U[i % k].add(verticesU[i])
            V[i % k].add(verticesV[i])
    return U, V


def get_key(item):
    """
    returns the sorting criteria for the edges. All edges are sorted from small to large values
    :param item: one item
    :return: returns the weight of the edge
    """
    return item[2]


def find_mst(U, V, E):
    """
    finds the mst of graph G = (U union V, E)
    :param U: vertices U
    :param V: vertices V
    :param E: edges of the graph
    :return: the mst and edges not in the mst of the graph
    """
    vertices = set()
    for v in V:
        vertices.add(v)
    for u in U:
        vertices.add(u)
    E = sorted(E, key=get_key)
    connected_component = set()
    mst = []
    remove_edges = set()
    while len(mst) < len(vertices) - 1 and len(connected_component) < len(vertices):
        for edge in E:
            if len(connected_component) == 0:
                connected_component.add(edge[0])
                connected_component.add(edge[1])
                mst.append(edge)
                E.remove(edge)
                break
            else:
                if edge[0] in connected_component:
                    if edge[1] in connected_component:
                        remove_edges.add(edge)
                        E.remove(edge)
                    else:
                        connected_component.add(edge[1])
                        mst.append(edge)
                        E.remove(edge)
                        break
                elif edge[1] in connected_component:
                    if edge[0] in connected_component:
                        remove_edges.add(edge)
                        E.remove(edge)
                    else:
                        connected_component.add(edge[0])
                        mst.append(edge)
                        E.remove(edge)
                        break
    for edge in E:
        remove_edges.add(edge)
    if len(mst) != len(vertices) - 1 or len(connected_component) != len(vertices):
        print("Error: MST found cannot be correct \n Length mst: ", len(mst), "\n Total connected vertices: ", len(connected_component), "\n Number of vertices: ", len(vertices))
        print("MST found: ", mst)
        quit()
    return mst, remove_edges


def get_edges(U, V, E):
    """
    :param U: subset of vertices (u_j)
    :param V: subset of vertices (v_i)
    :param E: all edges of the whole graph
    :return: all edges that are part of the graph u_j U v_j
    """

    edges = []
    for node1 in U:
        for node2 in V:
            if node2 in E[node1]:
                edges.append((node1, node2, E[node1][node2]))
    return U, V, edges


def reduce_edges(vertices, E, c, epsilon):
    """
    Uses PySpark to distribute the computation of the MSTs,
    Randomly partition the vertices twice in k subsets (U = {u_1, u_2, .., u_k}, V = {v_1, v_2, .., v_k})
    For every intersection between U_i and V_j, create the subgraph and find the MST in this graph
    Remove all edges from E that are not part of the MST in the subgraph
    :param vertices: vertices in the graph
    :param E: edges of the graph
    :param c: constant
    :param epsilon:
    :return:The reduced number of edges
    """
    conf = SparkConf().setAppName('MST_Algorithm')
    sc = SparkContext.getOrCreate(conf=conf)

    n = len(vertices)
    k = math.ceil(n**((c - epsilon) / 2))
    U, V = partion_vertices(vertices, k)
    rddU = sc.parallelize(U)
    rddV = sc.parallelize(V)
    rddUV = rddU.cartesian(rddV)
    rddUV1 = rddUV.map(lambda x: get_edges(x[0], x[1], E))
    rddUV2 = rddUV1.map(lambda x: (find_mst(x[0], x[1], x[2])))
    first = rddUV2.map(lambda x: x[0])
    second = rddUV2.map(lambda x: x[1])
    mst = first.collect()
    removed_edges = second.collect()
    return mst, removed_edges


def remove_edges(E, removed_edges, msts):
    """
    Removes the edges, which are removed when generating msts
    :param E: current edges
    :param removed_edges: edges to be removed
    :param msts: edges in the msts
    :return: return the updated edge dict
    """
    for removed_edge in removed_edges:
        for edge in removed_edge:
            if edge[1] in E[edge[0]]:
                del E[edge[0]][edge[1]]
            if edge[0] in E[edge[1]]:
                del E[edge[1]][edge[0]]
        for mst in msts:
            for edge in mst:
                if edge[1] not in E[edge[0]]:
                    E[edge[0]][edge[1]] = edge[2]
                if edge[0] not in E[edge[1]]:
                    E[edge[1]][edge[0]] = edge[2]
    return E


def create_mst(V, E, epsilon, m, size):
    """
    Creates the mst of the graph G = (V, E).
    As long as the number of edges is greater than n ^(1 + epsilon), the number of edges is reduced
    Then the edges that needs to be removed are removed from E and the size is updated.
    :param V: Vertices
    :param E: edges
    :param epsilon:
    :param m: number of machines
    :param size: number of edges
    :return: returns the reduced graph with at most np.power(n, 1 + epsilon) edges
    """
    n = len(V)
    c = math.log(m / n, n)
    while size > np.power(n, 1 + epsilon):
        mst, removed_edges = reduce_edges(V, E, c, epsilon)
        E = remove_edges(E, removed_edges, mst)
        size_removed_edges = 0
        for i in removed_edges:
            size_removed_edges += len(i)
        size = size - size_removed_edges
        c = (c - epsilon) / 2
    # Now the number of edges is reduced and can be moved to a single machine
    V = set(range(n))
    items = E.items() # returns [(x, {y : 1})]
    edges = []
    for item in items:
        items2 = item[1].items()
        for item2 in items2:
            edges.append((item[0], item2[0], item2[1]))
    mst, removed_edges = find_mst(V, V, edges)

    return mst


def plot_mst(vertices, mst):
    x = []
    y = []
    c = []
    area = []
    for i in range(len(vertices)):
        x.append(float(vertices[i][0]))
        y.append(float(vertices[i][1]))
        area.append(0.1)
        c.append("black")
    plt.scatter(x, y, c=c, s=area)

    for i in range(len(mst)):
        linex = [float(x[int(mst[i][0])])]
        liney = [float(y[int(mst[i][0])])]
        linex.append(float(x[int(mst[i][1])]))
        liney.append(float(y[int(mst[i][1])]))
        print(linex, liney, mst[i][2])
        plt.plot(linex, liney)
    plt.show()
    return


def main():
    """
    For every dataset, it creates the mst and plots the clustering
    """
    parser = ArgumentParser()
    parser.add_argument('--test', help="Used for smaller dataset and testing", action="store_true")
    parser.add_argument('--epsilon', help="epsilon [default=1/8]", type=float, default=1/8)
    parser.add_argument('--machines', help="Number of machines [default=1]", type=int, default=1)
    args = parser.parse_args()

    print("Start generating MST")
    if args.test:
        print("Test argument given")

    start_time = datetime.now()
    print("Starting time:", start_time)

    # create_mst()
    datasets = get_clustering_data()

    for dataset in datasets:
        timestamp = datetime.now()
        print("Start creating Distance Matrix...")
        dm, E, size, vertices_coordinates = create_distance_matrix(dataset[0][0])
        V = list(range(len(dm)))
        print("Size dataset: ", len(dm))
        print("Created distance matrix in: ", datetime.now() - timestamp)
        print("Start creating MST...")
        timestamp = datetime.now()
        mst = create_mst(V, E, epsilon=args.epsilon, m=args.machines, size=size)
        print("Created MST in: ", datetime.now() - timestamp)
        print("MST:\n", mst)
        print("Start creating plot of MST...")
        timestamp = datetime.now()
        plot_mst(dataset[0][0], mst)
        print("Created plot of MST in: ", datetime.now() - timestamp)


    print("Done...")


if __name__ == '__main__':
    # Initial call to main function
    main()
    quit()
    mst = []
    vertices = []
    with open('mst.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        lines = 0
        for row in csv_reader:
            line = (row[0], row[1], row[2])
            mst.append(line)

    with open('vertices.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        lines = 0
        for row in csv_reader:
            line = (row[0], row[1])
            vertices.append(line)

    plot_mst(vertices, mst)