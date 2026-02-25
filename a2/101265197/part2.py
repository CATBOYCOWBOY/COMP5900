import pandas as pd
import igraph as ig

# Git web dataset of Github users' follow relationships - https://snap.stanford.edu/data/github-social.html
def retrieve_git_web_dataset() -> ig.Graph:
    vertices = pd.read_csv('git_web_ml/musae_git_target.csv').set_index('id')
    edges = pd.read_csv('git_web_ml/musae_git_edges.csv')
    graph = ig.Graph(
        edges=list(edges.itertuples(index=False, name=None)),
        directed=False,
    )

    graph.vs['username'] = vertices['name'].tolist()

    return graph

# Dataset of Twitch gamers' follow relationships - https://snap.stanford.edu/data/twitch_gamers.html
def retrieve_twitch_dataset() -> ig.Graph:
    vertices = pd.read_csv('twitch_gamers/large_twitch_features.csv')
    edges = pd.read_csv('twitch_gamers/large_twitch_edges.csv')
    graph = ig.Graph(
        edges=list(edges.itertuples(index=False, name=None)),
        directed=False,
    )

    graph.vs['numeric_id'] = vertices['numeric_id'].tolist()

    return graph