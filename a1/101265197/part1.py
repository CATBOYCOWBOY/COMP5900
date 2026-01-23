import igraph as ig
import pandas as pd

def analyze_git_web_dataset() -> ig.Graph:
    vertices = pd.read_csv('git_web_ml/musae_git_target.csv').set_index('id')
    edges = pd.read_csv('git_web_ml/musae_git_edges.csv')
    graph = ig.Graph(
        edges=list(edges.itertuples(index=False, name=None)),
        directed=False,
    )

    graph.vs['username'] = vertices['name'].tolist()

    print("Github Web Dataset Graph Analysis:")
    print(graph.summary())

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

def compare_datasets(graph1: ig.Graph, graph2: ig.Graph):
    pass

if __name__ == "__main__":
    git_web_graph = analyze_git_web_dataset()
    twitch_graph = analyze_twitch_dataset()
    compare_datasets(git_web_graph, twitch_graph)