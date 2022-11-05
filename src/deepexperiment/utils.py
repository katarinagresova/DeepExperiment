from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable
import numpy as np
import random

@register_keras_serializable()
class ResBlock(layers.Layer):
    """
    Defines a Residual block. For more information refer to the original paper at https://arxiv.org/abs/1512.03385 .
    """

    def __init__(self, downsample=False, filters=16, kernel_size=3):

        super(ResBlock, self).__init__()

        # store parameters
        self.downsample = downsample
        self.filters = filters
        self.kernel_size = kernel_size

        # initialize inner layers
        self.conv1 = layers.Conv2D(kernel_size=self.kernel_size,
                                   strides=(1 if not self.downsample else 2),
                                   filters=self.filters,
                                   padding="same")
        self.activation1 = layers.ReLU()
        self.batch_norm1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(kernel_size=self.kernel_size,
                                   strides=1,
                                   filters=self.filters,
                                   padding="same")
        if self.downsample:
          self.conv3 = layers.Conv2D(kernel_size=1,
                                      strides=2,
                                      filters=self.filters,
                                      padding="same")

        self.activation2 = layers.ReLU()
        self.batch_norm2 = layers.BatchNormalization()

    def call(self, inputs):

        x = self.conv1(inputs)
        x = self.activation1(x)
        x = self.batch_norm1(x)
        x = self.conv2(x)

        if self.downsample:
            inputs = self.conv3(inputs)

        x = layers.Add()([inputs, x])

        x = self.activation2(x)
        x = self.batch_norm2(x)

        return x

def one_hot_encoding(miRNA, gene, tensor_dim=(50, 20, 1)):
    """
    fun encodes miRNAs and mRNAs in df into binding matrices
    :param df: dataframe containing 'gene' and 'miRNA' columns
    :param tensor_dim: output shape of the matrix
    :return: numpy array of predictions
    """
    # alphabet for watson-crick interactions.
    alphabet = {"AT": 1., "TA": 1., "GC": 1., "CG": 1., "AU": 1., "UA": 1.}
    # create empty main 2d matrix array
    N = 1  # number of samples in df
    shape_matrix_2d = (N, *tensor_dim)  # 2d matrix shape
    # initialize dot matrix with zeros
    ohe_matrix_2d = np.zeros(shape_matrix_2d, dtype="float32")

    # compile matrix with watson-crick interactions.
    #for index, row in df.iterrows():
    for bind_index, bind_nt in enumerate(gene.upper()):
        for mirna_index, mirna_nt in enumerate(miRNA.upper()):
            base_pairs = bind_nt + mirna_nt
            ohe_matrix_2d[0, bind_index, mirna_index, 0] = alphabet.get(base_pairs, 0)

    return ohe_matrix_2d

def one_hot_encoding_batch(df, tensor_dim=(50, 20, 1)):
    # alphabet for watson-crick interactions.
    alphabet = {"AT": 1., "TA": 1., "GC": 1., "CG": 1.}
    # labels to one hot encoding
    label = df["label"].to_numpy()
    # create empty main 2d matrix array
    N = df.shape[0]  # number of samples in df
    shape_matrix_2d = (N, *tensor_dim)  # 2d matrix shape
    # initialize dot matrix with zeros
    ohe_matrix_2d = np.zeros(shape_matrix_2d, dtype="float32")

    # compile matrix with watson-crick interactions.
    for index, row in df.iterrows():
        for bind_index, bind_nt in enumerate(row.gene.upper()):
            for mirna_index, mirna_nt in enumerate(row.miRNA.upper()):
                base_pairs = bind_nt + mirna_nt
                ohe_matrix_2d[index, bind_index, mirna_index, 0] = alphabet.get(base_pairs, 0)

    return ohe_matrix_2d, label

def get_indices(samples, model, compare_func):
  """ Returns the indices of the samples which prediction score satisfies the compare_func 
  """
  data, _ = one_hot_encoding_batch(samples)
  preds = model.predict(data)
  indices = []
  for i in range(len(preds)):
    if compare_func(preds[i]):
      indices.append(i)
  return indices
  
def get_true_positive_index(pos_samples, model):
  """ Returns the index of a random true positive sample. Computed based on the provided model's predictions. """
  if not hasattr(get_true_positive_index, 'indices'):
    get_true_positive_index.indices = get_indices(pos_samples, model, lambda x: x[1] > x[0])
  position = random.randrange(len(get_true_positive_index.indices))
  return get_true_positive_index.indices[position]

def get_false_negative_index(pos_samples, model):
  """ Returns the index of a random false negative sample. Computed based on the provided model's predictions. """
  if not hasattr(get_false_negative_index, 'indices'):
    get_false_negative_index.indices = get_indices(pos_samples, model, lambda x: x[0] > x[1])
  position = random.randrange(len(get_false_negative_index.indices))
  return get_false_negative_index.indices[position]

def get_true_negative_index(neg_samples, model):
  """ Returns the index of a random true negative sample. Computed based on the provided model's predictions. """
  if not hasattr(get_true_negative_index, 'indices'):
    get_true_negative_index.indices = get_indices(neg_samples, model, lambda x: x[0] > x[1])
  position = random.randrange(len(get_true_negative_index.indices))
  return get_true_negative_index.indices[position]

def get_false_positive_index(neg_samples, model):
  """ Returns the index of a random false positive sample. Computed based on the provided model's predictions. """
  if not hasattr(get_false_positive_index, 'indices'):
    get_false_positive_index.indices = get_indices(neg_samples, model, lambda x: x[1] > x[0])
  position = random.randrange(len(get_false_positive_index.indices))
  return get_false_positive_index.indices[position]