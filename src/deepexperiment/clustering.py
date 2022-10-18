from sklearn.cluster import AgglomerativeClustering
import tensorflow as tf

def cluster_attention(attentions, dist_function, n_clusters):
    dist = []
    for i in range(len(attentions)):
        row = []
        for j in range(len(attentions)):
            row.append(dist_function(attentions[i], attentions[j]))
        dist.append(row)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage='complete').fit(dist)
    return clustering.labels_

def compute_cluster_consensus(score, labels):
  cluster_consensus = []
  for num in range(len(set(labels))):
    cluster = score[labels == num].sum(axis=0)
    cluster = tf.maximum(cluster, 0) / tf.math.reduce_max(cluster)
    cluster_consensus.append(cluster)

  return cluster_consensus