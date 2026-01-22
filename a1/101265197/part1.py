import igraph as ig
import pandas as pd

def analyze_git_web_dataset() -> ig.Graph:
    targets_data = pd.read_csv('git_web_ml/musae_git_target.csv').set_index('id')
    edges_data = pd.read_csv('git_web_ml/musae_git_edges.csv')
    graph = ig.Graph(
        edges=list(edges_data.itertuples(index=False, name=None)),
        directed=False,
    )

    graph.vs['username'] = targets_data['name'].tolist()

    return graph

def analyze_twitch_dataset() -> ig.Graph:
    vertices = pd.read_csv('twitch_ml/musae_twitch_target.csv')
    edges = pd.read_csv('twitch_ml/musae_twitch_edges.csv')
    return ig.Graph.TupleList(edges.values.tolist(), directed=False)

def compare_datasets(graph1: ig.Graph, graph2: ig.Graph):
    pass