import math
from copy import deepcopy
from datetime import datetime
from pyspark import SparkConf, SparkContext
from sklearn.datasets import make_circles, make_moons, make_blobs, make_swiss_roll, make_s_curve
from sklearn.neighbors import KDTree
from argparse import ArgumentParser

from Plotter import *
from DataReader import *


def get_clustering_data():
    """
    Retrieves all toy datasets from sklearn
    :return: circles, moons, blobs datasets.
    """
    n_samples = 1500
    noisy_circles = make_circles(n_samples=n_samples, factor=.5,
                                 noise=0.05)
    noisy_moons = make_moons(n_samples=n_samples, noise=0.05)
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

    swiss_roll = make_swiss_roll(n_samples, noise=0.05)

    s_shape = make_s_curve(n_samples, noise=0.05)

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
        (no_structure, {}),
        (swiss_roll, {}),
        (s_shape, {})]

    return datasets


def map_contract_graph(_lambda, leader):
    def contraction(adj):
        u, nu = adj
        c, v = u, u
        S = []
        while v not in S:
            S.append(v)
            c = min(c, v)
            v = _lambda[v]
        # c = min(c, v)
        A = list(filter(lambda e: leader[e[0]] != c, nu))
        return c, A
    return contraction


def reduce_contract_graph(leader):
    def reduce_contraction(Nu, A):
        for v, w in A:
            l = leader[v]
            new = True
            for i, e in enumerate(Nu):
                if l == e[0]:
                    new = False
                    Nu[i] = (l, min(w, e[1]))
            if new:
                Nu.append((l, w))
        return Nu

    return reduce_contraction


def find_best_neighbours(adj):
    u, nu = adj
    nn = u
    if len(nu) > 0:
        min_v, min_w = nu[0]
        for v, w in nu:
            if w < min_w:
                min_v, min_w = v, w
        nn = min_v
    return u, nn


def find_leader(_lambda):
    def find(adj):
        u, nu = adj
        c, v = u, u
        S = []
        prev = 0
        cnt = 0
        while v not in S:
            S.append(v)
            if v is None:
                print(u, nu)
                print('cnt: ', cnt)
                print('prev', prev)
                print('lambda', _lambda[prev])
                print('lambda', _lambda)
                quit()
            prev = v
            v = _lambda[v]
            cnt += 1
            c = min(c, v)
        return u, c

    return find


def affinity_clustering(adj, vertex_coordinates, plot_intermediate, num_clusters=3):
    conf = SparkConf().setAppName('MST_Algorithm')
    sc = SparkContext.getOrCreate(conf=conf)
    clusters = [[i] for i in range(len(adj))]
    yhats = []
    graph = deepcopy(adj)
    rdd = sc.parallelize(adj)

    i = 0
    imax = 20
    while i < imax:
        if len(graph) <= num_clusters:
            break
        num_edges = sum(map(lambda v: len(v[1]), graph))
        if num_edges == 0:
            break

        rdd1 = rdd.map(find_best_neighbours).collect()
        _lambda = [None] * len(adj)
        for line in rdd1:
            _lambda[line[0]] = line[1]

        # Find leader
        leader = [None] * len(adj)
        rdd1 = rdd.map(find_leader(_lambda)).collect()
        for line in rdd1:
            leader[line[0]] = line[1]

        for j in range(len(adj)):
            l = leader[j]
            if l is not None and not l == j:
                clusters[l].extend(clusters[j])
                clusters[j].clear()

        yhat = [None] * len(adj)
        for c, cluster in enumerate(clusters):
            for v in cluster:
                yhat[v] = c
        yhats.append(yhat)

        # Contraction
        rdd = (rdd.map(map_contract_graph(_lambda=_lambda, leader=leader))
               .foldByKey([], reduce_contract_graph(leader)))
        graph = rdd.collect()
        i += 1

    return i, graph, yhats


def get_nearest_neighbours(V, k=5, leaf_size=2, buckets=False):
    def get_sort_key(item):
        return item[1]

    if buckets:
        adj = []
        for key in V:
            nu = []
            sorted_list = sorted(V[key].items(), key=get_sort_key)
            last = -1
            to_shuffle = []
            for i in range(k):
                if last != sorted_list[i][1]:
                    random.shuffle(to_shuffle)
                    for item in to_shuffle:
                        nu.append(item)
                    to_shuffle = []
                else:
                    to_shuffle.append((sorted_list[i][0], sorted_list[i][1]))
                last = sorted_list[i][1]

            random.shuffle(to_shuffle)
            for item in to_shuffle:
                nu.append(item)
            adj.append((key, nu))

    else:
        kd_tree = KDTree(V, leaf_size=leaf_size)
        dist, ind = kd_tree.query(V, k=k + 1)

        adj = []
        for i in range(len(V)):
            nu = []
            for j in range(1, len(dist[i])):
                nu.append((ind[i][j], dist[i][j]))
            adj.append((i, nu))
    return adj


# num buckets = log_(1 + beta) (W)
def create_buckets(E, alpha, beta, W):
    num_buckets = math.ceil(math.log(W, (1 + beta)))
    buckets = []
    prev_end = 0
    for i in range(num_buckets):
        now_end = np.power((1 + beta), i) + (np.random.uniform(-alpha, alpha) * np.power((1 + beta), i))
        if i < num_buckets - 1:
            buckets.append((prev_end, now_end))
            prev_end = now_end
        else:
            buckets.append((prev_end, W + 0.00001))

    for key in E:
        for edge in E[key]:
            bucket_number = 1
            for bucket in buckets:
                if bucket[0] <= E[key][edge] < bucket[1]:
                    E[key][edge] = bucket_number
                    break
                bucket_number += 1
    return E, buckets


def shift_edge_weights(E, gamma=0.01):
    max_weight = 0
    for key in E:
        for edge in E[key]:
            # TODO: fix shift (remove np.random.uniform(0, 100) +)
            E[key][edge] = np.random.uniform(0, 100) + max(E[key][edge] * np.random.uniform(-gamma, gamma), 0)
            max_weight = max(E[key][edge], max_weight)
    return E, max_weight


def main():
    parser = ArgumentParser()
    parser.add_argument('--test', help='Used for smaller dataset and testing', action='store_true')
    parser.add_argument('--epsilon', help='epsilon [default=1/8]', type=float, default=1 / 8)
    parser.add_argument('--machines', help='Number of machines [default=1]', type=int, default=1)
    parser.add_argument('--buckets', help='Use buckets [default=False]', type=bool, default=False)
    args = parser.parse_args()

    print('Start generating MST')
    if args.test:
        print('Test argument given')

    start_time = datetime.now()
    print('Starting time:', start_time)

    file_location = 'Results/test/'
    plotter = Plotter(None, None, file_location)

    # Read data
    data_reader = DataReader()
    loc = 'datasets/CA-AstroPh.txt'
    V, size, E = data_reader.read_data_set_from_txtfile(loc)
    print('Read dataset: ', loc)

    # Toy datasets
    datasets = get_clustering_data()

    timestamp = datetime.now()
    for dataset in datasets:
        k = 1499
        if args.buckets:
            beta = 0.5  # 0 <= beta <= 1 (buckets)
            alpha = 0.1  # shift of buckets
            gamma = 0.001  # shift of edge weights

            E, size, vertex_coordinates, W = data_reader.create_distance_matrix(dataset[0][0], full_dm=True)
            E, W = shift_edge_weights(E, gamma)
            E, buckets = create_buckets(E, alpha, beta, W)
            k = len(vertex_coordinates) - 1
            adjacency_list = get_nearest_neighbours(E, k, buckets=True)
        else:
            adjacency_list = get_nearest_neighbours(dataset[0][0], k)

        runs, graph, yhats = affinity_clustering(adjacency_list, vertex_coordinates=None, plot_intermediate=False)
        print('Graph size: ', len(graph), graph)
        print('Runs: ', runs)
        print('yhats: ', yhats[runs - 1])

        plotter.plot_yhats(yhats, dataset[0][0])


if __name__ == '__main__':
    # Initial call to main function
    main()
