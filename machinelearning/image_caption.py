"""Main module."""

from .idx_to_label import IDX_TO_LABEL
import tensorflow as tf


def raw_to_model_output(model, image):
    image = tf.keras.applications.vgg19.preprocess_input(image, data_format=None)
    return model.predict(image)


def predict_image_class(model, image, idx_to_label=IDX_TO_LABEL, k=1):
    output_idx = raw_to_model_output(model, image)
    result = tf.math.top_k(output_idx[0], k=k)
    result = list(zip(result.indices.numpy(), result.values.numpy()))
    result = [(idx_to_label[index], value) for index, value in result]
    return result
