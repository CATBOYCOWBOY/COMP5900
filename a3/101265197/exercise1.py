import igraph as ig
from sklearn.metrics import adjusted_mutual_info_score


def community_ecg(g, t=16, min_weight=0.05):
    """
    Ensemble Clustering for Graphs (ECG) - Poulin & Théberge (2019).

    Runs Louvain t times, weights each edge by how often its endpoints
    were in the same community, then runs a final Louvain on the
    weighted graph.

    Parameters:
        g          : igraph Graph
        t          : ensemble size (number of Louvain runs)
        min_weight : minimum edge weight (for edges never co-assigned)

    Returns:
        VertexClustering from the final weighted Louvain run
    """
    co_counts = [0] * g.ecount()

    for _ in range(t):
        partition = g.community_multilevel()
        membership = partition.membership
        for i, edge in enumerate(g.es):
            if membership[edge.source] == membership[edge.target]:
                co_counts[i] += 1

    weights = [min_weight + (1 - min_weight) * (c / t) for c in co_counts]
    g.es['weight'] = weights

    return g.community_multilevel(weights='weight')


g = ig.Graph.Famous('Zachary')

# Ground truth: two factions from Zachary (1977), 0-indexed
# ground_truth[i] == 0 => Mr. Hi's group, 1 => Officer John A's group
# Sourced from Wikipedia
ground_truth = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1]

ecg = community_ecg(g)

print(f"Communities found: {len(ecg)}")                                                                                                                                                     


ami = adjusted_mutual_info_score(ground_truth, ecg.membership)
print(f"{'Algorithm':<20} {'AMI Score':>10}")
print("-" * 31)
print(f"{'ECG':<20} {ami:>10.4f}")
