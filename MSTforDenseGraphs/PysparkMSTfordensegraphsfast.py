import pandas as pd
import scipy.spatial
import os
import sys

from argparse import ArgumentParser
from datetime import datetime
from sklearn.datasets import make_circles, make_moons, make_blobs, make_swiss_roll, make_s_curve

from Plotter import *
from DataReader import *
from reduceEdges import *
from helperFunctions import *

os.environ['PYSPARK_PYTHON'] = ".venv/Scripts/python.exe"

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

def create_distance_matrix(dataset):
    """
    Creates the distance matrix for a dataset with only vertices. Also adds the edges to a dict.
    :param dataset: dataset without edges
    :return: distance matrix, a dict of all edges and the total number of edges
    """
    vertices = []
    size = 0
    three_d = False
    for line in dataset:
        if len(line) == 2:
            vertices.append([line[0], line[1]])
        elif len(line) == 3:
            vertices.append([line[0], line[1], line[2]])
            three_d = True
    if three_d:
        dict = {}
        for i in range(len(dataset)):
            dict2 = {}
            for j in range(i + 1, len(dataset)):
                dict2[j] = np.sqrt(np.sum(np.square(dataset[i] - dataset[j])))
                size += 1
            dict[i] = dict2

    else:
        d_matrix = scipy.spatial.distance_matrix(vertices, vertices, threshold=1000000)
        dict = {}
        # Run with less edges
        for i in range(len(d_matrix)):
            dict2 = {}
            for j in range(i, len(d_matrix)):
                if i != j:
                    size += 1
                    dict2[j] = d_matrix[i][j]
            dict[i] = dict2
    return dict, size, vertices

def remove_edges(E, removed_edges):
    """
    Removes the edges, which are removed when generating msts
    :param E: current edges
    :param removed_edges: edges to be removed
    :return: return the updated edge dict
    """
    for edge in removed_edges:
        if edge[1] in E[edge[0]]:
            del E[edge[0]][edge[1]]
    return E


def create_mst(V, E, epsilon, size, vertex_coordinates, plot_intermediate=False, plotter=None,
               func_type: FunctionType = FunctionType.FULL_SPARK):
    """
    Creates the mst of the graph G = (V, E).
    As long as the number of edges is greater than n ^(1 + epsilon), the number of edges is reduced
    Then the edges that needs to be removed are removed from E and the size is updated.
    :param func_type: which internal function to call
    :param plotter: class to plot graphs
    :param V: Vertices
    :param E: edges
    :param epsilon:
    :param size: number of edges
    :param plot_intermediate: boolean to indicate if intermediate steps should be plotted
    :param vertex_coordinates: coordinates of vertices
    :return: returns the reduced graph with at most np.power(n, 1 + epsilon) edges
    """
    n = len(V)
    c = math.log(size / n, n)
    print("C", c)
    total_runs = 0
    while size > np.power(n, 1 + epsilon):
        total_runs += 1
        if plotter is not None:
            plotter.next_round()
        mst, removed_edges = reduce_edges(V, E, c, epsilon, func_type)
        if plot_intermediate and plotter is not None:
            if len(vertex_coordinates[0]) > 2:
                plotter.plot_mst_3d(mst, intermediate=True, plot_cluster=False, plot_num_machines=1)
            else:
                plotter.plot_mst_2d(mst, intermediate=True, plot_cluster=False, plot_num_machines=1)
        E = remove_edges(E, removed_edges)
        print('Total edges removed in this iteration', len(removed_edges))
        size = size - len(removed_edges)
        print('New total of edges: ', size)
        c = (c - epsilon) / 2
    # Now the number of edges is reduced and can be moved to a single machine
    V = set(range(n))
    items = E.items()  # returns [(x, {y : 1})]
    edges = []
    for item in items:
        items2 = item[1].items()
        for item2 in items2:
            edges.append((item[0], item2[0], item2[1]))
    mst, removed_edges = find_mst(V, V, edges)
    print("#####\n\nTotal runs: ", total_runs, "\n\n#####")
    return mst


def main():
    """
    For every dataset, it creates the mst and plots the clustering
    """
    parser = ArgumentParser()
    parser.add_argument('--test', help='Used for smaller dataset and testing', action='store_true')
    parser.add_argument('--epsilon', help='epsilon [default=1/8]', type=float, default=1 / 8)
    parser.add_argument('--machines', help='Number of machines [default=1]', type=int, default=1)
    parser.add_argument('--reduce_edge_function', help='Which function to use to reduce edges', type=str, default='FULL_SPARK')
    parser.add_argument('--names_datasets', help='Names of the datasets', type=str, nargs='+')
    parser.add_argument('--num_clusters', help='Number of clusters for each dataset', type=int, nargs='+')
    args = parser.parse_args()

    print('Start generating MST')
    if args.test:
        print('Test argument given')

    start_time = datetime.now()
    print('Starting time:', start_time)

    time = []
    file_location = 'MSTforDenseGraphs/Results_buckets/dense_algorithm/'
    plotter = Plotter(None, None, file_location)
    function_type = FunctionType[args.reduce_edge_function]
    # function_type = FunctionType.FULL_SPARK

    # names_datasets = ['2d-20c-no0'] #, '2sp2glob', '3-spiral', 'D31', 'spiralsquare', 'square1', 'twenty', 'fourty']
    # extra_datasets = ['cluto-t7-10k', 'cluto-t8-8k', 'complex8', 'complex9']
    names_datasets = args.names_datasets
    num_clusters = args.num_clusters
    datasets = [pd.read_csv(f'MSTforDenseGraphs/datasets/{name}.csv', header=None).to_numpy() for name in names_datasets]

    # num_clusters = [20, 4, 3, 9, 10, 8, 9, 31, 6, 4, 20, 40]
    size_and_timings = []
    cnt = 0
    for dataset in datasets:
        start_time = datetime.now()
        if cnt < 0:
            cnt += 1
            continue
        timestamp = datetime.now()
        print('Start creating Distance Matrix...')
        E, size, vertex_coordinates = create_distance_matrix(dataset)
        plotter.set_vertex_coordinates(vertex_coordinates)
        plotter.set_dataset(names_datasets[cnt])
        plotter.update_string()
        plotter.reset_round()
        V = list(range(len(vertex_coordinates)))
        print('Size dataset: ', len(vertex_coordinates))
        dataset_size = len(vertex_coordinates)
        edge_size = size
        print('Created distance matrix in: ', datetime.now() - timestamp)
        print('Start creating MST...')
        timestamp = datetime.now()
        start_mst_time = datetime.now()
        mst = create_mst(V, E, epsilon=args.epsilon, size=size, vertex_coordinates=vertex_coordinates,
                         plot_intermediate=True, plotter=plotter, func_type=function_type)
        end_mst_time = datetime.now()
        dataset_mst_time = end_mst_time - start_mst_time
        print('Found MST in: ', datetime.now() - timestamp)
        time.append(datetime.now() - timestamp)
        print('Start creating plot of MST...')
        timestamp = datetime.now()
        print(len(vertex_coordinates[0]))
        if len(vertex_coordinates[0]) > 2:
            plotter.plot_mst_3d(mst, intermediate=False, plot_cluster=True, num_clusters=num_clusters[cnt])
        else:
            plotter.plot_mst_2d(mst, intermediate=False, plot_cluster=True, num_clusters=num_clusters[cnt])
        print('Created plot of MST in: ', datetime.now() - timestamp)
        end_time = datetime.now()
        print(';'.join([str(x) for x in [names_datasets[cnt], dataset_size, edge_size, dataset_mst_time, str(end_time - start_time)]]),
        file=sys.stderr)
        cnt += 1

if __name__ == '__main__':
    # Initial call to main function
    main()
