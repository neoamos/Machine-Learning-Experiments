
import tensorflow as tf
from tensorflow import keras

def l1_ssim_edge(l1_weight=1.0, ssim_weight=1.0, edge_weight=1.0):

  def func(y_true, y_pred):
    ssim_error = ssim_loss(y_true, y_pred)

    edge_error = edge_loss(y_true, y_pred)

    mae = tf.keras.losses.MeanAbsoluteError()
    mae_error = mae(y_true, y_pred)

    return l1_weight * mae_error  + edge_weight * edge_error + ssim_weight * ssim_error

  return func


def ssim_loss(y_true, y_pred):
  y_pred = tf.expand_dims(y_pred, 3)
  y_true = tf.expand_dims(y_true, 3)
  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def edge_loss(y_true, y_pred):
  y_pred = tf.expand_dims(y_pred, 3)
  y_true = tf.expand_dims(y_true, 3)
  edges_pred = tf.image.sobel_edges(y_pred)
  edges_true = tf.image.sobel_edges(y_true)

  # dx_true, dy_true = edges_true[:, :, :, :, 0], edges_true[:, :, :, :, 1]
  # dx_pred, dy_pred = edges_pred[:, :, :, :, 0], edges_pred[:, :, :, :, 1]

  return tf.reduce_mean(tf.abs(edges_pred - edges_true))

