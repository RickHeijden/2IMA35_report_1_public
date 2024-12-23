import random


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

def get_edges(U, V, E):
    """
    :param U: subset of vertices (u_j)
    :param V: subset of vertices (v_i)
    :param E: all edges of the whole graph
    :return: all edges that are part of the graph u_j U v_j
    """
    edges = set()
    for node1 in U:
        for node2 in V:
            if node1 in E:
                if node2 in E[node1]:
                    edges.add((node1, node2, E[node1][node2]))
                elif node2 in E:
                    if node1 in E[node2]:
                        edges.add((node2, node1, E[node2][node1]))
            elif node2 in E:
                if node1 in E[node2]:
                    edges.add((node2, node1, E[node2][node1]))
    edge_list = []
    for edge in edges:
        edge_list.append(edge)
    return U, V, edge_list

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
        if len(E) == 0:
            break
        change = False
        i = 0
        while i < len(E):
            if len(connected_component) == 0:
                connected_component.add(E[i][0])
                connected_component.add(E[i][1])
                mst.append(E[i])
                change = True
                E.remove(E[i])
                break
            else:
                if E[i][0] in connected_component:
                    if E[i][1] in connected_component:
                        remove_edges.add(E[i])
                        E.remove(E[i])
                    else:
                        connected_component.add(E[i][1])
                        mst.append(E[i])
                        E.remove(E[i])
                        change = True
                        break
                elif E[i][1] in connected_component:
                    if E[i][0] in connected_component:
                        remove_edges.add(E[i])
                        E.remove(E[i])
                    else:
                        connected_component.add(E[i][0])
                        mst.append(E[i])
                        E.remove(E[i])
                        change = True
                        break
                else:
                    i += 1
        if not change:
            if len(E) != 0:
                connected_component.add(E[0][0])
                connected_component.add(E[0][1])
                mst.append(E[0])
                E.remove(E[0])
    for edge in E:
        remove_edges.add(edge)
    if len(mst) != len(vertices) - 1 or len(connected_component) != len(vertices):
        print('Warning: parition cannot have a full MST! Missing edges to create full MST.')
        # print('Error: MST found cannot be correct \n Length mst: ', len(mst), '\n Total connected vertices: ',
        #       len(connected_component), '\n Number of vertices: ', len(vertices))
    return mst, remove_edges
