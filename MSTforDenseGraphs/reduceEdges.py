import math
import itertools
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

from MSTforDenseGraphs.helperFunctions import *
from enum import Enum


FunctionType = Enum("FunctionType", [
    "FULL_SPARK",
    "PURE_PYTHON",
    "INTERMEDIATE_COLLECT",
    "SINGLE_PARTITION",
    "DATAFRAMES",
    "CONFIG",
    "SINGLE_FUNCTION",
    "SINGLE_SLICE",
])


def reduce_edges(vertices, E, c, epsilon, func: FunctionType):
    """
    Uses PySpark to distribute the computation of the MSTs,
    Randomly partition the vertices twice in k subsets (U = {u_1, u_2, .., u_k}, V = {v_1, v_2, .., v_k})
    For every intersection between U_i and V_j, create the subgraph and find the MST in this graph
    Remove all edges from E that are not part of the MST in the subgraph
    :param vertices: vertices in the graph
    :param E: edges of the graph
    :param c: constant
    :param epsilon:
    :param func: which internal function to call
    :return:The reduced number of edges
    """
    n = len(vertices)
    k = math.ceil(n ** ((c - epsilon) / 2))
    print("k: ", k)
    U, V = partion_vertices(vertices, k)

    both: list[tuple] = []
    if func == FunctionType.FULL_SPARK:
        both = __reduce_edges_full_spark(U, V, E)
    elif func == FunctionType.PURE_PYTHON:
        both = __reduce_edges_pure_python(U, V, E)
    elif func == FunctionType.INTERMEDIATE_COLLECT:
        both = __reduce_edges_intermediate_collect(U, V, E)
    elif func == FunctionType.SINGLE_PARTITION:
        both = __reduce_edges_single_partition(U, V, E)
    elif func == FunctionType.DATAFRAMES:
        both = __reduce_edges_with_dataframes(U, V, E)
    elif func == FunctionType.CONFIG:
        both = __reduce_edges_with_config(U, V, E)
    elif func == FunctionType.SINGLE_FUNCTION:
        both = __reduce_edges_single_function(U, V, E)
    elif func == FunctionType.SINGLE_SLICE:
        both = __reduce_edges_single_slice(U, V, E)

    mst = []
    removed_edges = set()
    for i in range(len(both)):
        mst.append(both[i][0])
        for edge in both[i][1]:
            removed_edges.add(edge)

    return mst, removed_edges

def __reduce_edges_full_spark(U, V, E) -> list[tuple]:
    conf = SparkConf().setAppName('MST_Algorithm').setMaster('local[*]')
    conf = conf.set('spark.driver.memory', '2g')
    conf = conf.set('spark.executor.memory', '2g')
    sc = SparkContext.getOrCreate(conf=conf)
    sc.setLogLevel('OFF')

    rddUV = sc.parallelize(U).cartesian(sc.parallelize(V)).map(lambda x: get_edges(x[0], x[1], E)).map(
        lambda x: (find_mst(x[0], x[1], x[2])))
    both = rddUV.collect()

    sc.stop()
    return both

def __reduce_edges_single_slice(U, V, E) -> list[tuple]:
    conf = SparkConf().setAppName('MST_Algorithm').setMaster('local[*]')
    conf = conf.set('spark.driver.memory', '2g')
    conf = conf.set('spark.executor.memory', '2g')
    sc = SparkContext.getOrCreate(conf=conf)
    sc.setLogLevel('OFF')

    rddUV = sc.parallelize(U, 1).cartesian(sc.parallelize(V, 1)).map(lambda x: get_edges(x[0], x[1], E)).map(
        lambda x: (find_mst(x[0], x[1], x[2])))
    both = rddUV.collect()

    sc.stop()
    return both

def __reduce_edges_with_dataframes(U, V, E) -> list[tuple]:
    conf = SparkConf().setAppName('MST_Algorithm').setMaster('local[*]')
    conf = conf.set('spark.driver.memory', '2g')
    conf = conf.set('spark.executor.memory', '2g')
    sc = SparkSession.builder.config(conf=conf).getOrCreate()

    newU = []
    for u in U:
        newU.append(list(u))
    U = newU
    newV = []
    for v in V:
        newV.append(list(v))
    V = newV
    cartesian = list(itertools.product(U, V))

    rddUV = (
        sc.createDataFrame(cartesian).rdd
        .map(lambda x: get_edges(x[0], x[1], E), True)
        .map(lambda x: (find_mst(x[0], x[1], x[2])), True)
    )
    both = rddUV.collect()

    sc.stop()
    return both

def __reduce_edges_with_config(U, V, E) -> list[tuple]:
    conf = SparkConf().setAppName('MST_Algorithm').setMaster('local[1]')
    conf = conf.set("spark.executor.cores", "1")
    conf = conf.set("spark.executor.instances", "1")
    conf = conf.set("spark.dynamicAllocation.enabled", "false")
    conf = conf.set("spark.shuffle.compress", "false")
    conf = conf.set("spark.rdd.compress", "false")
    conf = conf.set("spark.default.parallelism", "1")
    conf = conf.set("spark.sql.shuffle.partitions", "1")
    conf = conf.set('spark.driver.memory', '2g')
    conf = conf.set('spark.executor.memory', '2g')
    sc = SparkContext.getOrCreate(conf=conf)
    sc.setLogLevel('OFF')

    rddUV = sc.parallelize(U).cartesian(sc.parallelize(V)).map(lambda x: get_edges(x[0], x[1], E)).map(
        lambda x: (find_mst(x[0], x[1], x[2])))
    both = rddUV.collect()

    sc.stop()
    return both

def __reduce_edges_single_partition(U, V, E) -> list[tuple]:
    conf = SparkConf().setAppName('MST_Algorithm').setMaster('local[*]')
    conf = conf.set('spark.log.level', 'OFF')
    conf = conf.set('spark.driver.memory', '2g')
    conf = conf.set('spark.executor.memory', '2g')
    sc = SparkContext.getOrCreate(conf=conf)
    sc.setLogLevel('OFF')

    rddUV = sc.parallelize(U, 1).cartesian(sc.parallelize(V, 1)).repartition(1).map(lambda x: get_edges(x[0], x[1], E), True).map(
        lambda x: (find_mst(x[0], x[1], x[2])), True)
    both = rddUV.collect()

    sc.stop()
    return both

def __reduce_edges_single_function(U, V, E) -> list[tuple]:
    conf = SparkConf().setAppName('MST_Algorithm').setMaster('local[*]')
    conf = conf.set('spark.driver.memory', '2g')
    conf = conf.set('spark.executor.memory', '2g')
    sc = SparkContext.getOrCreate(conf=conf)
    sc.setLogLevel('OFF')

    def cartesian_to_mst(x, E):
        edges = get_edges(x[0], x[1], E)
        return find_mst(edges[0], edges[1], edges[2])

    rddUV = sc.parallelize(U).cartesian(sc.parallelize(V)).map(lambda x: cartesian_to_mst(x, E), True)
    both = rddUV.collect()

    sc.stop()
    return both

def __reduce_edges_intermediate_collect(U, V, E) -> list[tuple]:
    conf = SparkConf().setAppName('MST_Algorithm').setMaster('local[*]')
    conf = conf.set('spark.driver.memory', '2g')
    conf = conf.set('spark.executor.memory', '2g')
    sc = SparkContext.getOrCreate(conf=conf)
    sc.setLogLevel('OFF')

    rdd_u_v = sc.parallelize(U).cartesian(sc.parallelize(V)).collect()
    rdd_edges = sc.parallelize(rdd_u_v).map(lambda x: get_edges(x[0], x[1], E)).collect()
    rdd_mst = sc.parallelize(rdd_edges).map(lambda x: (find_mst(x[0], x[1], x[2])))
    both = rdd_mst.collect()

    sc.stop()
    return both

def __reduce_edges_pure_python(U, V, E) -> list[tuple]:
    result = list(itertools.product(U, V))
    result = list(map(lambda x: get_edges(x[0], x[1], E), result))
    both = list(map(lambda x: (find_mst(x[0], x[1], x[2])), result))

    return both
