import igraph as ig
from partition_igraph import community_ecg
from sklearn.metrics import adjusted_mutual_info_score

g = ig.Graph.Famous('Zachary')

# Ground truth: two factions from Zachary (1977), 0-indexed
# ground_truth[i] == 0 => Mr. Hi's group, 1 => Officer John A's group
# Sourced from Wikipedia https://en.wikipedia.org/wiki/Zachary%27s_karate_club
ground_truth = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1]

print("Asessing communities found with community detection algorithms:")
print("-" * 31)

print(f"{'Algorithm':<20} {'AMI Score':>10}")
print("-" * 31)

# -----------------------------------------------
# ECG community detection:
# I tried default ensemble size from textbook (see: p150), but increasing the ensemble size seems to marginally improve
# AMI
ecg = community_ecg(g, ens_size=32)

ami = adjusted_mutual_info_score(ground_truth, ecg.membership)
print(f"{'ECG':<20} {ami:>10.4f}")

# -----------------------------------------------
# Leiden community detection:

# I think modularity is better in this situation since we know ground truth only has 2 communities
# We're not missing out on any small communities by optimizing for edge density within community
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

