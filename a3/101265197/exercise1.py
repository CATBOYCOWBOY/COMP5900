import igraph as ig
import networkx as nx
from partition_igraph import community_ecg
from sklearn.metrics import adjusted_mutual_info_score

import matplotlib
matplotlib.use('qt5agg')

import matplotlib.pyplot as plt

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

results = []

def record(name, partition):
    ami = adjusted_mutual_info_score(ground_truth, partition.membership)
    print(f"{name:<20} {ami:>10.4f}")
    results.append((name, ami))

# -----------------------------------------------
# ECG community detection:
# I tried default ensemble size of 16, but increasing the ensemble size seems to marginally improve
# AMI
ecg = community_ecg(g, ens_size=32)
record('ECG', ecg)

# -----------------------------------------------
# Leiden community detection:

# I think modularity is better in this situation since we know ground truth only has 2 communities
# We're not missing out on any small communities by optimizing for edge density within community
# Lowered resolution also improves AMI somewhat
leiden_partition = g.community_leiden(
    objective_function="modularity",
    resolution=0.6
)
record('Leiden', leiden_partition)

# -----------------------------------------------
# Louvain community detection:

louvain_partition = g.community_multilevel()
record('Louvain', louvain_partition)

# -----------------------------------------------
# Infomap community detection:

# differing trial counts seems to have no further returns past 10 trials (default). Reduced trial counts negatively 
# impact clustering effectiveness
infomap_partition = g.community_infomap()
record('Infomap', infomap_partition)

# -----------------------------------------------
# Label propagation community detection:

label_partition = g.community_label_propagation()
record('Label propagation', label_partition)

# -----------------------------------------------
# Girvan-newman community detection:

gn_partition = g.community_edge_betweenness().as_clustering(n=2)
record('Girvan-Newman', gn_partition)

# -----------------------------------------------
# CNM community detection:

cnm_partition = g.community_fastgreedy().as_clustering(n=2)
record('CNM', cnm_partition)

# -----------------------------------------------
# Plot results:

names, scores = zip(*results)
plt.figure(figsize=(9, 5))
bars = plt.bar(names, scores)
plt.bar_label(bars, fmt='%.4f', padding=3)
plt.ylim(0, 1)
plt.ylabel('AMI Score')
plt.title("Community Detection Algorithms' vs. Karate Club Community Ground Truth (AMI)")
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('./outputs/exercise1-ami-scores.png', dpi=150)
plt.show()

