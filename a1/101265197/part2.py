import numpy as np
import igraph as ig
import math
from part1 import retrieve_git_web_dataset
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
    degrees = source.degree()
    avg_deg = sum(degrees) / len(degrees)

    # worked this out assuming area of 1 and this radius formula generates ~= mean degree on both graphs (github and twitch)
    # although there is variation of small margin (less than .1 avg degree), the expected value of a vertex's degree should
    # always be avg_deg
    radius = math.sqrt(avg_deg / (math.pi * source.vcount()))
    generated_graph = ig.Graph.GRG(n=source.vcount(), radius=radius, torus=True)
    return generated_graph

# comparisons between the Git Web dataset and our generated graphs
# I chose this one instead of the Twitch dataset because the vertex degrees were strongly right skewed when plotted out
# which I found interesting compared to the more normal distribution of the Twitch dataset degrees
if __name__ == "__main__":
    print("\n----------------------------------------\n")
    print("Comparison of Git Web Graph (Graph 1) to Various Random Graph Models (Graph 2)\n")
    git_web_graph = retrieve_git_web_dataset()

    g_np = g_np_graph(git_web_graph)
    print("Git Web Graph vs Erdos-Renyi G_n,p Graph Analysis:")
    compare_graphs(git_web_graph, g_np)
    print("----------------------------------------")

    g_nm = g_nm_graph(git_web_graph)
    print("Git Web Graph vs Erdos-Renyi G_n,m Graph Analysis:")
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