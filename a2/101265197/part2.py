import pandas as pd
import igraph as ig
import networkx as nx

# Git web dataset of Github users' follow relationships - https://snap.stanford.edu/data/github-social.html
# Note - unlinke the graphs in part 1, these graphs seem to follow the edge/vertex counts outlined in SNAP,
# seemingly validating my theory that the mismatch is in normalizing vertex names.
def retrieve_git_web_dataset() -> ig.Graph:
    vertices = pd.read_csv('undirected/git_web_ml/musae_git_target.csv').set_index('id')
    edges = pd.read_csv('undirected/git_web_ml/musae_git_edges.csv')
    graph = ig.Graph(
        edges=list(edges.itertuples(index=False, name=None)),
        directed=False,
    )

    graph.vs['username'] = vertices['name'].tolist()

    return graph

# Dataset of Twitch gamers' follow relationships - https://snap.stanford.edu/data/twitch_gamers.html
def retrieve_twitch_dataset() -> ig.Graph:
    vertices = pd.read_csv('undirected/twitch_gamers/large_twitch_features.csv')
    edges = pd.read_csv('undirected/twitch_gamers/large_twitch_edges.csv')
    graph = ig.Graph(
        edges=list(edges.itertuples(index=False, name=None)),
        directed=False,
    )

    graph.vs['numeric_id'] = vertices['numeric_id'].tolist()

    return graph

def deg_assortativity(graph: ig.Graph) -> float:
    return graph.assortativity_degree(directed=False)

def richclub_coefficient(graph: ig.Graph) -> list:
    nx_graph = graph.to_networkx()
    # It looks weird but these coefficients are correct - high degree vertices pretty effictively form a complete graph on both graphs used
    # and if we only print the top 50 degree counts in terms of their coefficients, we're going to see all 1s imo.
    rc = nx.rich_club_coefficient(nx_graph, normalized=False)
    return sorted(rc.items(), key=lambda x: x[0])

def print_single_graph_stats(graph: ig.Graph, name: str):
    import numpy as np
    print(f"{name} graph stats:")
    print(f"-----------------------------------------\n")

    print(f"Vertex count: {graph.vcount()}")
    print(f"Edge count: {graph.ecount()}")
    print(f"Number of degree 0 vertices: {np.sum(np.array(graph.degree()) == 0)}")
    print(f"Degree assortativity: {deg_assortativity(graph)}")

    print(f"\nRich-club coefficients (degree threshold, coefficient):")
    for threshold, coefficient in richclub_coefficient(graph)[-50:]:
        print(f"  k={threshold}: {coefficient:.4f}")

    print("\n\n\n")

def print_graph_information():
    git_web_ds = retrieve_git_web_dataset()
    twitch_ds = retrieve_twitch_dataset()
    print_single_graph_stats(git_web_ds, "GitHub social")
    print_single_graph_stats(twitch_ds, "Twitch gamers")

if __name__ == "__main__":
    print_graph_information()