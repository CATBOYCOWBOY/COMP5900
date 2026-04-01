import igraph as ig
import networkx as nx
from partition_igraph import community_ecg
from sklearn.metrics import adjusted_mutual_info_score

g = ig.Graph.Famous('Zachary')

# Ground truth: factions from Zachary (1977), 0-indexed
# ground_truth[i] == 0 => Mr. Hi's group, 1 => Officer John A's group
# Sourced from networkx's karate_club_graph(), which follows the original paper
_nx_karate = nx.karate_club_graph()
ground_truth = [0 if _nx_karate.nodes[n]['club'] == 'Mr. Hi' else 1 for n in sorted(_nx_karate.nodes())]

print("Asessing communities found with community detection algorithms:")
print("-" * 31)

print(f"{'Algorithm':<20} {'AMI Score':>10}")
print("-" * 31)

# -----------------------------------------------
# ECG community detection:
# I tried default ensemble size of 16, but increasing the ensemble size seems to marginally improve
# AMI
ecg = community_ecg(g, ens_size=32)

ami = adjusted_mutual_info_score(ground_truth, ecg.membership)
print(f"{'ECG':<20} {ami:>10.4f}")

# -----------------------------------------------
# Leiden community detection:

# I think modularity is better in this situation since we know ground truth only has 2 communities
# We're not missing out on any small communities by optimizing for edge density within community
# Lowered resolution also improves AMI somewhat
leiden_partition = g.community_leiden(
    objective_function="modularity",
    resolution=0.6
)

ami = adjusted_mutual_info_score(ground_truth, leiden_partition.membership)
print(f"{'Leiden':<20} {ami:>10.4f}")

# -----------------------------------------------
# Louvain community detection:

louvain_partition = g.community_multilevel()

ami = adjusted_mutual_info_score(ground_truth, louvain_partition.membership)
print(f"{'Louvain':<20} {ami:>10.4f}")

# -----------------------------------------------
# Infomap community detection:

# differing trial counts seems to have no further returns past 10 trials (default)
infomap_partition = g.community_infomap()

ami = adjusted_mutual_info_score(ground_truth, infomap_partition.membership)
print(f"{'Infomap':<20} {ami:>10.4f}")

# -----------------------------------------------
# Label propagation community detection:

label_partition = g.community_label_propagation()

ami = adjusted_mutual_info_score(ground_truth, label_partition.membership)
print(f"{'Label propagation':<20} {ami:>10.4f}")

# -----------------------------------------------
# Girvan-newman community detection:

gn_partition = g.community_edge_betweenness().as_clustering(n=2)

ami = adjusted_mutual_info_score(ground_truth, gn_partition.membership)
print(f"{'Girvan-newman':<20} {ami:>10.4f}")

# -----------------------------------------------
# CNM community detection:

cnm_partition = g.community_fastgreedy().as_clustering(n=2)

ami = adjusted_mutual_info_score(ground_truth, cnm_partition.membership)
print(f"{'CNM':<20} {ami:>10.4f}")

