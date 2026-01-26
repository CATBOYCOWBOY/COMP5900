import numpy as np
import igraph as ig
import math
from part1 import analyze_git_web_dataset
from part1 import compare_graphs

def g_np_graph(source: ig.Graph) -> ig.Graph:
    vertices = source.vcount()
    density = source.ecount() / (vertices * (vertices - 1) / 2)
    return ig.Graph.Erdos_Renyi(n=vertices, p=density)

def g_nm_graph(source: ig.Graph) -> ig.Graph:
    vertices = source.vcount()
    edges = source.ecount()
    return ig.Graph.Erdos_Renyi(n=vertices, m=edges)

def k_regular_graph(source: ig.Graph) -> ig.Graph:
    vertices = source.vcount()
    avg_degree = int(np.round(np.average(source.degree())))
    if avg_degree % 2 != 0:
        avg_degree += 1 
    return ig.Graph.K_Regular(n=vertices, k=avg_degree)

def degree_sequence_graph(source: ig.Graph) -> ig.Graph:
    degree_sequence = source.degree()
    return ig.Graph.Degree_Sequence(degree_sequence, method="vl")

def random_geometric_graph(source: ig.Graph) -> ig.Graph:
    avg_deg = np.average(source.degree())

    # working backwards from the probability that 2 points land within radius r in a unit square of each other
    # where the initial point is minimum r distance from an edge we can find r when the probability (density)
    # and count of points is known
    # it should(?) also work out for a torus which is what we are meant to be using in this question
    radius = math.sqrt(avg_deg / math.pi)
    return ig.Graph.GRG(n=source.vcount(), radius=radius, torus=True)

if __name__ == "__main__":
    git_web_graph = analyze_git_web_dataset()

    g_np = g_np_graph(git_web_graph)
    print("Git Web Graph vs Erdos-Renyi G(n, p) Graph Analysis:")
    compare_graphs(git_web_graph, g_np)
    print("----------------------------------------")

    g_nm = g_nm_graph(git_web_graph)
    print("Git Web Graph vs Erdos-Renyi G(n, m) Graph Analysis:")
    compare_graphs(git_web_graph, g_nm)
    print("----------------------------------------")

    k_reg = k_regular_graph(git_web_graph)
    print("Git Web Graph vs k-Regular Graph Analysis:")
    compare_graphs(git_web_graph, k_reg)
    print("----------------------------------------")

    deg_seq = degree_sequence_graph(git_web_graph)
    print("Git Web Graph vs Degree Sequence Graph Analysis:")
    compare_graphs(git_web_graph, deg_seq)
    print("----------------------------------------")

    rand_geo = random_geometric_graph(git_web_graph)
    print("Git Web Graph vs Random Geometric Graph Analysis:")
    compare_graphs(git_web_graph, rand_geo)