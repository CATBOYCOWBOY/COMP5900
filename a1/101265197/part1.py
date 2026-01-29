import igraph as ig
import pandas as pd
import math
import statistics
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Important - this code was written with Python 3.12.12 in mind, with the following packages installed 
# (specified in included requirements.txt):

# igraph v1.0.0
# pandas v3.0.0
# matplotlib v3.10.8
# numpy v2.4.1

# Please use Python 3.12.x (other versions of Python I tested are not guaranteed to have tkinter compatibility, 
# especially if you are using homebrew) to ensure that python-tk@3.12 can work with the matplotlib TkAgg backend for 
# displaying histograms. 
#
# You can create a virtual environment and install the packages using the env's pip
# 
# If you do not want to use python 3.12, you can remove the matplotlib.use('TkAgg') line, 
# either replace it with matplotlib.use('Qt5Agg') or another interactive MatPltoLib backend of your choice that is 
# compatible with your system.

HISTOGRAM_BAR_WIDTH = 0.35


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

def generate_degree_histogram(graph1: ig.Graph, graph2: ig.Graph):
    deg1 = graph1.degree()
    deg2 = graph2.degree()

    # Setting bin count via Sturge's rule: k = 1 + log2(n)
    n = max(len(deg1), len(deg2))
    num_bins = int(1 + math.log2(n))
    min_deg = max(1, min(min(deg1), min(deg2)))  # Avoid log(0)
    max_deg = max(max(deg1), max(deg2))

    # binning based on log scale, helps with the right tail distribution we see in graphs.
    # otherwise the histogram looks like 1 bar on the left and nothing else
    log_min = math.log10(min_deg)
    log_max = math.log10(max_deg)
    bin_edges = [10 ** (log_min + i * (log_max - log_min) / num_bins) for i in range(num_bins + 1)]
    bin_edges = [int(edge) for edge in bin_edges] 
    bin_edges = list(dict.fromkeys(bin_edges)) # Remove duplicates
    bin_edges[-1] = max_deg + 1
    num_bins = len(bin_edges) - 1

    # Counts for each bin, per graph
    hist1, _ = pd.cut(deg1, bins=bin_edges, include_lowest=True, retbins=True)
    hist2, _ = pd.cut(deg2, bins=bin_edges, include_lowest=True, retbins=True)
    counts1 = hist1.value_counts().sort_index().values
    counts2 = hist2.value_counts().sort_index().values

    # Graph configs
    bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(num_bins)]
    x = range(num_bins)
    bar_width = HISTOGRAM_BAR_WIDTH

    _, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - bar_width/2 for i in x], counts1, bar_width, label='Graph 1', color='blue')
    ax.bar([i + bar_width/2 for i in x], counts2, bar_width, label='Graph 2', color='red')

    ax.set_xlabel('Degree Range')
    ax.set_ylabel('Number of Vertices')
    ax.set_title('Degree Distribution Histogram (Refer to stdout/terminal for Graph Names)')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.show()

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
    generate_degree_histogram(graph1, graph2)

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
