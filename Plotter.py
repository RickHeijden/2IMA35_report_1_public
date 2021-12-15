import matplotlib.pyplot as plt


def get_key(item):
    """
    returns the sorting criteria for the edges. All edges are sorted from small to large values
    :param item: one item
    :return: returns the weight of the edge
    """
    return item[2]


def create_clusters(clusters, dict_edges):
    i = 0
    while i < len(clusters):
        pop = False
        for j in range(i):
            if clusters[i][0] in clusters[j]:
                clusters.pop(i)
                pop = True
                break
        if pop:
            continue

        todo = []
        for j in range(clusters[i][0]):
            if j in dict_edges:
                if clusters[i][0] in dict_edges[j]:
                    clusters[i].append(j)
                    todo.append(j)
        if clusters[i][0] in dict_edges:
            for j in range(len(dict_edges[clusters[i][0]])):
                todo.append(dict_edges[clusters[i][0]][j])
                clusters[i].append(dict_edges[clusters[i][0]][j])

        while len(todo) > 0:
            first = todo.pop()
            for k in range(first):
                if k in dict_edges:
                    if first in dict_edges[k] and k not in clusters[i]:
                        clusters[i].append(k)
                        todo.append(k)
            if first in dict_edges:
                for k in range(len(dict_edges[first])):
                    if dict_edges[first][k] not in clusters[i]:
                        clusters[i].append(dict_edges[first][k])
                        todo.append(dict_edges[first][k])
        i += 1
    for i in range(len(clusters)):
        clusters[i] = sorted(clusters[i])

    return clusters


class Plotter:

    def __init__(self, vertex_coordinates, name_dataset, file_loc):
        self.vertex_coordinates = vertex_coordinates
        self.name_dataset = name_dataset
        self.file_loc = file_loc
        self.round = 0
        self.colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'darkorange', 'dodgerblue', 'deeppink', 'khaki', 'purple',
                       'springgreen', 'tomato', 'slategray', 'forestgreen', 'mistyrose', 'mediumorchid',
                       'rebeccapurple', 'lavender', 'cornflowerblue', 'lightseagreen', 'brown']
        self.machine_string = "{}_round_{}_machine_".format(self.name_dataset, self.round)

    def update_string(self):
        self.machine_string = "{}_round_{}_machine_".format(self.name_dataset, self.round)

    def set_dataset(self, name_dataset):
        self.name_dataset = name_dataset

    def set_vertex_coordinates(self, vertex_coordinates):
        self.vertex_coordinates = vertex_coordinates

    def set_file_loc(self, file_loc):
        self.file_loc = file_loc

    def reset_round(self):
        self.round = 0

    def next_round(self):
        self.round += 1
        self.update_string()

    def plot_mst_3d(self, mst, intermediate=False, plot_cluster=False, plot_num_machines=0, num_clusters=2):
        x = []
        y = []
        z = []
        c = []
        area = []

        for i in range(len(self.vertex_coordinates)):
            x.append(float(self.vertex_coordinates[i][0]))
            y.append(float(self.vertex_coordinates[i][1]))
            z.append(float(self.vertex_coordinates[i][2]))
            area.append(0.1)
            c.append('k')

        if intermediate:
            if plot_num_machines > 0:
                cnt = 0
                for m in mst:
                    ax = plt.axes(projection='3d')
                    ax.scatter3D(x, y, z, c=c, s=area)
                    ax.view_init(azim=75, elev=5)

                    for i in range(len(m)):
                        linex = [float(x[int(m[i][0])]), float(x[int(m[i][1])])]
                        liney = [float(y[int(m[i][0])]), float(y[int(m[i][1])])]
                        linez = [float(z[int(m[i][0])]), float(z[int(m[i][1])])]
                        ax.plot(linex, liney, linez, self.colors[cnt])

                    cnt = (cnt + 1) % len(self.colors)
                    filename = self.file_loc + self.machine_string + '{}'.format(cnt)
                    plt.savefig(filename, dpi='figure')
                    plt.clf()
                    if cnt >= plot_num_machines:
                        break

            cnt = 0
            ax = plt.axes(projection='3d')
            ax.scatter3D(x, y, z, c=c, s=area)
            ax.view_init(azim=75, elev=5)
            for m in mst:
                for i in range(len(m)):
                    linex = [float(x[int(m[i][0])]), float(x[int(m[i][1])])]
                    liney = [float(y[int(m[i][0])]), float(y[int(m[i][1])])]
                    linez = [float(z[int(m[i][0])]), float(z[int(m[i][1])])]
                    ax.plot(linex, liney, linez, self.colors[cnt])
                cnt = (cnt + 1) % len(self.colors)
            filename = self.file_loc + self.machine_string + 'all'
            plt.savefig(filename, dpi='figure')
            plt.clf()

        elif plot_cluster:
            ax = plt.axes(projection='3d')
            ax.scatter3D(x, y, z, c=c, s=area)
            ax.view_init(azim=75, elev=5)

            edges = sorted(mst, key=get_key, reverse=True)
            removed_edges = []
            clusters = []
            for i in range(num_clusters - 1):
                edge = edges.pop(0)
                removed_edges.append(edge)
                clusters.append([edge[0]])
                clusters.append([edge[1]])
                linex = [float(x[edge[0]]), float(x[edge[1]])]
                liney = [float(y[edge[0]]), float(y[edge[1]])]
                linez = [float(z[edge[0]]), float(z[edge[1]])]
                ax.plot(linex, liney, linez, "k")

            dict_edges = dict()
            for edge in edges:
                if edge[0] in dict_edges:
                    dict_edges[edge[0]].append(edge[1])
                else:
                    dict_edges[edge[0]] = [edge[1]]

            clusters = create_clusters(clusters, dict_edges)

            x_cluster = []
            y_cluster = []
            z_cluster = []
            c_cluster = []
            area_cluster = []

            for i in range(len(clusters)):
                for vertex in clusters[i]:
                    x_cluster.append(float(self.vertex_coordinates[vertex][0]))
                    y_cluster.append(float(self.vertex_coordinates[vertex][1]))
                    z_cluster.append(float(self.vertex_coordinates[vertex][2]))
                    area_cluster.append(0.2)
                    c_cluster.append(self.colors[i])
            ax.scatter3D(x_cluster, y_cluster, z_cluster, c=c_cluster, s=area_cluster)

            for i in range(len(mst)):
                if mst[i] in removed_edges:
                    continue
                linex = [float(x[int(mst[i][0])]), float(x[int(mst[i][1])])]
                liney = [float(y[int(mst[i][0])]), float(y[int(mst[i][1])])]
                linez = [float(z[int(mst[i][0])]), float(z[int(mst[i][1])])]
                for j in range(len(clusters)):
                    if mst[i][0] in clusters[j]:
                        ax.plot3D(linex, liney, linez, c=self.colors[j])
            filename = self.file_loc + self.name_dataset + '_clusters'
            plt.savefig(filename, dpi='figure')
            plt.clf()
        else:
            ax = plt.axes(projection='3d')
            ax.scatter3D(x, y, z, c=c, s=area)
            ax.view_init(azim=75, elev=5)

            for i in range(len(mst)):
                linex = [float(x[int(mst[i][0])]), float(x[int(mst[i][1])])]
                liney = [float(y[int(mst[i][0])]), float(y[int(mst[i][1])])]
                linez = [float(z[int(mst[i][0])]), float(z[int(mst[i][1])])]
                ax.plot3D(linex, liney, linez)
            filename = self.file_loc + self.name_dataset + '_final'
            plt.savefig(filename, dpi='figure')
            plt.clf()

    def plot_mst_2d(self, mst, intermediate=False, plot_cluster=False, plot_num_machines=0, num_clusters=2):
        x = []
        y = []
        c = []
        area = []

        for i in range(len(self.vertex_coordinates)):
            x.append(float(self.vertex_coordinates[i][0]))
            y.append(float(self.vertex_coordinates[i][1]))
            area.append(0.1)
            c.append('k')

        if intermediate:
            if plot_num_machines > 0:
                cnt = 0
                for m in mst:
                    plt.scatter(x, y, c=c, s=area)

                    for i in range(len(m)):
                        linex = [float(x[int(m[i][0])]), float(x[int(m[i][1])])]
                        liney = [float(y[int(m[i][0])]), float(y[int(m[i][1])])]
                        plt.plot(linex, liney, self.colors[cnt])

                    cnt = (cnt + 1) % len(self.colors)
                    filename = self.file_loc + self.machine_string + '{}'.format(cnt)
                    plt.savefig(filename, dpi='figure')
                    plt.clf()
                    if cnt >= plot_num_machines:
                        break

            cnt = 0
            for m in mst:
                for i in range(len(m)):
                    linex = [float(x[int(m[i][0])]), float(x[int(m[i][1])])]
                    liney = [float(y[int(m[i][0])]), float(y[int(m[i][1])])]
                    plt.plot(linex, liney, self.colors[cnt])
                cnt = (cnt + 1) % len(self.colors)
            filename = self.file_loc + self.machine_string + 'all'
            plt.savefig(filename, dpi='figure')
            plt.clf()
        elif plot_cluster:
            edges = sorted(mst, key=get_key, reverse=True)
            removed_edges = []
            clusters = []
            for i in range(num_clusters - 1):
                edge = edges.pop(0)
                removed_edges.append(edge)
                clusters.append([edge[0]])
                clusters.append([edge[1]])
                linex = [float(x[edge[0]]), float(x[edge[1]])]
                liney = [float(y[edge[0]]), float(y[edge[1]])]
                plt.plot(linex, liney, "k")

            dict_edges = dict()
            for edge in edges:
                if edge[0] in dict_edges:
                    dict_edges[edge[0]].append(edge[1])
                else:
                    dict_edges[edge[0]] = [edge[1]]

            clusters = create_clusters(clusters, dict_edges)

            x_cluster = []
            y_cluster = []
            c_cluster = []
            area_cluster = []

            for i in range(len(clusters)):
                for vertex in clusters[i]:
                    x_cluster.append(float(self.vertex_coordinates[vertex][0]))
                    y_cluster.append(float(self.vertex_coordinates[vertex][1]))
                    area_cluster.append(0.2)
                    c_cluster.append(self.colors[i])
            plt.scatter(x_cluster, y_cluster, c=c_cluster, s=area_cluster)

            for i in range(len(mst)):
                if mst[i] in removed_edges:
                    continue
                linex = [float(x[int(mst[i][0])]), float(x[int(mst[i][1])])]
                liney = [float(y[int(mst[i][0])]), float(y[int(mst[i][1])])]
                for j in range(len(clusters)):
                    if mst[i][0] in clusters[j]:
                        plt.plot(linex, liney, c=self.colors[j])
            filename = self.file_loc + self.name_dataset + '_clusters'
            plt.savefig(filename, dpi='figure')
            plt.clf()
        else:
            for i in range(len(mst)):
                linex = [float(x[int(mst[i][0])]), float(x[int(mst[i][1])])]
                liney = [float(y[int(mst[i][0])]), float(y[int(mst[i][1])])]
                plt.plot(linex, liney)
            filename = self.file_loc + self.name_dataset + '_final'
            plt.savefig(filename, dpi='figure')
            plt.clf()
