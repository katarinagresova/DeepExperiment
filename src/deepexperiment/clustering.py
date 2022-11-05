import tensorflow as tf

def compute_cluster_consensus(score, labels):
  cluster_consensus = []
  for num in range(len(set(labels))):
    cluster = score[labels == num].sum(axis=0)
    cluster = tf.maximum(cluster, 0) / tf.math.reduce_max(cluster)
    cluster_consensus.append(cluster)

  return cluster_consensus