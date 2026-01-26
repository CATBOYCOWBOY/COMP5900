import igraph as ig
import pandas as pd
import math
import statistics

def analyze_git_web_dataset() -> ig.Graph:
    vertices = pd.read_csv('git_web_ml/musae_git_target.csv').set_index('id')
    edges = pd.read_csv('git_web_ml/musae_git_edges.csv')
    graph = ig.Graph(
        edges=list(edges.itertuples(index=False, name=None)),
        directed=False,
    )

    graph.vs['username'] = vertices['name'].tolist()

    return graph

def analyze_twitch_dataset() -> ig.Graph:
    vertices = pd.read_csv('twitch_gamers/large_twitch_features.csv')
    edges = pd.read_csv('twitch_gamers/large_twitch_edges.csv')
    graph = ig.Graph(
        edges=list(edges.itertuples(index=False, name=None)),
        directed=False,
    )

    graph.vs['numeric_id'] = vertices['numeric_id'].tolist()

    return graph

def compare_graphs(graph1: ig.Graph, graph2: ig.Graph):
    deg1 = graph1.degree()
    deg2 = graph2.degree()
    print("\nGraph Comparison:")
    print("----------------------------------------\n")
    print("Graph 1 maximum degree: ", max(deg1))
    print("Graph 2 maximum degree: ", max(deg2))
    print("\nGraph 1 average degree: ", sum(deg1) / len(deg1))
    print("Graph 2 average degree: ", sum(deg2) / len(deg2))
    print("\nMedian degree of Graph 1: ", statistics.median(deg1))
    print("Median degree of Graph 2: ", statistics.median(deg2))
    print("\nGraph 1 density: ", sum(deg1) / math.comb(graph1.vcount(), 2))
    print("Graph 2 density: ", sum(deg2) / math.comb(graph2.vcount(), 2))
    print("\nGraph 1 average clustering coefficient: ", graph1.transitivity_undirected())
    print("Graph 2 average clustering coefficient: ", graph2.transitivity_undirected())

if __name__ == "__main__":
    git_web_graph = analyze_git_web_dataset()

    print("rozemberczki2019multiscale Github Web Dataset Graph Analysis (Graph 1):")
    print(f"  Number of vertices: {git_web_graph.vcount()}")
    print(f"  Number of edges: {git_web_graph.ecount()}")
    print(f"  Number of degree-0 vertices: {sum(1 for deg in git_web_graph.degree() if deg == 0)}")

    print("\n----------------------------------------\n")

    twitch_graph = analyze_twitch_dataset()

    print("rozemberczki2021twitch Dataset Graph Analysis: (Graph 2):")
    print(f"  Number of vertices: {twitch_graph.vcount()}")
    print(f"  Number of edges: {twitch_graph.ecount()}")
    print(f"  Number of degree-0 vertices: {sum(1 for deg in twitch_graph.degree() if deg == 0)}")

    compare_graphs(git_web_graph, twitch_graph)
