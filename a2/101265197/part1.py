import igraph as ig
import numpy as np

# This script was written with python 3.13.x in mind. It will likely require python version 3.12 or above to work
# out of the box.

def retrieve_reddit_dataset() -> ig.Graph:
    edges = []
    with open("directed/soc-redditHyperlinks-body.tsv") as f:
        next(f)  # skip header
        for line in f:
            parts = line.split("\t")
            edges.append((parts[0], parts[1]))

    vertices = list(set(v for edge in edges for v in edge))
    vertex_to_id = {v: i for i, v in enumerate(vertices)}

    graph = ig.Graph(directed=True)
    graph.add_vertices(len(vertices))
    graph.vs["name"] = vertices
    graph.add_edges([(vertex_to_id[s], vertex_to_id[t]) for s, t in edges])
    return graph

def retrieve_wiki_dataset() -> ig.Graph:
    edges = []
    with open("directed/wiki-RfA.txt") as f:
        src = None
        for line in f:
            line = line.strip()
            if line.startswith("SRC:"):
                src = line[4:]
            elif line.startswith("TGT:"):
                tgt = line[4:]
                # This is a check to make sure that we're actually getting a real user's vote
                # There are many instances of (see: line 5657) unnamed voters issuing votes, and this results in
                # an unnamed voter being marked as very central. 
                # Since we don't know if the unnamed votes come from one or multiple users, I am electing to exclude
                # them from calculations.
                if src and tgt:
                    edges.append((src, tgt))

    vertices = list(set(v for edge in edges for v in edge))
    vertex_to_id = {v: i for i, v in enumerate(vertices)}

    graph = ig.Graph(directed=True)
    graph.add_vertices(len(vertices))
    graph.vs["name"] = vertices
    graph.add_edges([(vertex_to_id[s], vertex_to_id[t]) for s, t in edges])
    return graph

def degree_centrality_top_50(graph: ig.Graph) -> list[str]:
    n = graph.vcount()

    in_centrality = [d / (n - 1) for d in graph.degree(mode="in")]
    out_centrality = [d / (n - 1) for d in graph.degree(mode="out")]

    centrality_pairs = zip(in_centrality, out_centrality)
    centrality = [(d1 + d2) / 2 for d1, d2 in centrality_pairs]

    top_50_indices = np.argsort(centrality)[-50:][::-1]
    return [graph.vs[i]["name"] for i in top_50_indices]

def eigenvector_centrality_top_50(graph: ig.Graph) -> list[str]:
    centrality = graph.eigenvector_centrality()

    top_50_indices = np.argsort(centrality)[-50:][::-1]
    return [graph.vs[i]["name"] for i in top_50_indices]

def pagerank_centrality_top_50(graph: ig.Graph, damping: float) -> list[str]:
    centrality = graph.personalized_pagerank(damping)

    top_50_indices = np.argsort(centrality)[-50:][::-1]
    return [graph.vs[i]["name"] for i in top_50_indices]

def hub_score_top_50(graph: ig.Graph) -> list[str]:
    score = graph.hub_score()

    top_50_indices = np.argsort(score)[-50:][::-1]
    return [graph.vs[i]["name"] for i in top_50_indices]


def authority_score_top_50(graph: ig.Graph) -> list[str]:
    score = graph.authority_score()

    top_50_indices = np.argsort(score)[-50:][::-1]
    return [graph.vs[i]["name"] for i in top_50_indices]

def print_graph_stats(graph: ig.Graph, name: str):
    print(f"{name} graph stats:")
    print(f"-----------------------------------------\n")

    print(f"Vertex count: {graph.vcount()}")
    print(f"Edge count: {np.sum(graph.degree(mode="out"))}")
    print(f"Number of degree 0 vertices: {np.sum(np.array(graph.degree()) == 0)}")

    print(f"-----------------------------------------\n")
    print(f"Centrality measures:")
    print(f"-----------------------------------------\n")

    print(f"Top 50 vertices in degree centrality:")
    for idx, vertex in enumerate(degree_centrality_top_50(graph)):
        print(f"{idx + 1}. {vertex}")
    
    print(f"\n-----------------------------------------\n")

    print(f"Top 50 vertices in eigenvector centrality:")
    for idx, vertex in enumerate(eigenvector_centrality_top_50(graph)):
        print(f"{idx + 1}. {vertex}")
    
    print(f"\n-----------------------------------------\n")

    print(f"Top 50 vertices in pagerank centrality (no damping):")
    for idx, vertex in enumerate(pagerank_centrality_top_50(graph, 0.0)):
        print(f"{idx + 1}. {vertex}")
    
    print(f"\n-----------------------------------------\n")

    print(f"Top 50 vertices in pagerank centrality (damping of 0.85):")
    for idx, vertex in enumerate(pagerank_centrality_top_50(graph, 0.85)):
        print(f"{idx + 1}. {vertex}")
    
    print(f"\n-----------------------------------------\n")

    print(f"Top 50 vertices in Kleinberg's hub score:")
    for idx, vertex in enumerate(hub_score_top_50(graph)):
        print(f"{idx + 1}. {vertex}")
    
    print(f"\n-----------------------------------------\n")

    print(f"Top 50 vertices in Kleinberg's authority score:")
    for idx, vertex in enumerate(authority_score_top_50(graph)):
        print(f"{idx + 1}. {vertex}")
    
    
    print("\n\n\n")

if __name__ == "__main__":
    print_graph_stats(retrieve_reddit_dataset(), "Reddit hyperlink")
    print_graph_stats(retrieve_wiki_dataset(), "Wikipedia admin requests")