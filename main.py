import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn import cluster
from sklearn.manifold import TSNE


# Parse args
parser = argparse.ArgumentParser()
parser.add_argument('--k', help='Number of clusters in K-Means', type=int)
args = parser.parse_args()


# Read embeddings
embeddings = np.memmap('data/transr_FB15K237.bin', dtype='float32', mode='r')
embeddings = embeddings.reshape((14541, 128))

# Read id-to-name mapping
id2name = {}
with open('data/FB15K237_id2name.txt') as file:
    n = int(file.readline())
    for i in range(n):
        content = file.readline()
        content = content.strip().split()
        id, name = content[0], ' '.join(content[1:])
        id2name[int(id)] = name

# Read graphs
edges = []
with open('data/FB15K237_train2id.txt', 'r') as file:
    n_edges = int(file.readline())
    for i in range(n_edges):
        content = file.readline()
        h, t, r = content.strip().split()
        edges.append((int(h), int(t)))
graph = nx.Graph(edges)

# K-means
n_clusters = args.k
kmeans_model = cluster.KMeans(n_clusters=n_clusters, max_iter=100, init='k-means++')
kmeans_model.fit(embeddings)
cluster_labels = kmeans_model.labels_

# Create and analyze sub-graphs of every cluster
topk = 10
for i in range(n_clusters):
    print('Cluster {}:'.format(i))
    subgraph_indices = set(np.where(cluster_labels == i)[0])
    print(' > {} items in total'.format(len(subgraph_indices)))
    subgraph_edges = []
    for h, t in edges:
        if h in subgraph_indices or t in subgraph_indices:
            subgraph_edges.append((h, t))
    subgraph = nx.Graph(subgraph_edges)
    degree_centrality = nx.degree_centrality(subgraph)
    degree_centrality = dict(filter(lambda item: item[0] in subgraph_indices, degree_centrality.items()))
    ranks = dict(sorted(degree_centrality.items(), key=lambda item: -item[1]))
    keys = list(ranks.keys())
    print(' > Ranks:')
    for k in range(min(topk, len(ranks))):
        print('   > Top {}: {} (id: {}, score: {})'.format(k + 1, id2name[keys[k]], keys[k], degree_centrality[keys[k]]))

# T-SNE (a way for dimension reduction)
# t_sne = TSNE()
# t_sne.fit(embeddings)
