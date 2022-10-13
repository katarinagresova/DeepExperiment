import tensorflow as tf
from .utils import ResBlock
import cv2 
import numpy as np
import shap

class GradCam():
    def __init__(self, model):
        self.model = model

    def __call__(self, input, pred_index=None, last_conv_layer_name=None):
        if last_conv_layer_name == 'all':
            return self._make_all_layer_gradcam(input, pred_index)

        if last_conv_layer_name is None:
            last_conv_layer_name = self._get_last_conv_layer_name()

        return self._make_gradcam(input, pred_index, last_conv_layer_name)

    def _make_gradcam(self, input, pred_index, last_conv_layer_name):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
            [self.model.inputs], [self.model.get_layer(last_conv_layer_name).output, self.model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(input)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the output class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def _make_all_layer_gradcam(self, input, pred_index):
        layers = [layer.name for layer in reversed(self.model.layers) if len(layer.output_shape) == 4 and (layer.__class__.__name__ == 'ReLU' or isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, ResBlock))]
        cams = []
        for layer in layers:
            cam = self._make_gradcam(input, pred_index, layer)
            cam = cv2.resize(cam, (input.shape[2], input.shape[1]))
            cams.append(cam)
        return np.mean(cams, axis=0)

    def _get_last_conv_layer_name(self):
        layers = [layer.name for layer in reversed(self.model.layers) if len(layer.output_shape) == 4 and (isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, ResBlock))]
        return layers[0]

class DeepShap():
    def __init__(self, model, background):
        self.model = model
        self.background = background
        self.e = shap.DeepExplainer(self.model, self.background)
        # these layers are not supported, so the workaround presented on the github is to use passthrough
        shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough 
        shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough

    def __call__(self, input):  
        shap_values = self.e.shap_values(input)
        return shap_values[0], shap_values[1]