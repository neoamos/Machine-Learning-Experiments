
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

def l1_ssim_edge(l1_weight=1.0, ssim_weight=1.0, edge_weight=1.0, l=tf.keras.losses.MeanAbsoluteError()):

  def func(y_true, y_pred):
    ssim_error = ssim_loss(y_true, y_pred)
    edge_error = edge_loss(y_true, y_pred)
    l_error = l(y_true, y_pred)

    return l1_weight * l_error  + edge_weight * edge_error + ssim_weight * ssim_error

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


def percent_relative_error(threshold):
  """
  Calculates the percent of pixes that are within threshold relative error
  """
  def relative_error(y_true, y_pred):
    error = tf.math.abs(y_true-y_pred)
    relative_error = error/y_true
    # tf.print(tf.math.reduce_max(relative_error))
    # tf.print(tf.math.reduce_min(relative_error))

    less = tf.math.less(relative_error, threshold)

    return tf.reduce_mean(tf.cast(less, tf.float32))

  relative_error.__name__ = "relative_error_{}".format(threshold*100)
  return relative_error


def log_mse(y_true, y_pred):
  mse = tf.keras.losses.MeanSquaredError()

  return mse(tf.math.log(y_true), tf.math.log(y_pred))

def scale_invariant_mse(y_true, y_pred):
  y_true = tf.math.log(y_true)
  y_pred = tf.math.log(y_pred)
  mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)

  n = tf.math.pow(tf.size(y_true), 2)
  tf.print(y_true.shape)

  scale_factor = y_true-y_pred
  scale_factor = tf.math.pow(tf.reduce_sum(scale_factor), 2)
  
  return mse - scale_factor


def rmse(y_true, y_pred):
  y_true = y_true * 10
  y_pred = y_pred * 10

  mse = tf.keras.losses.MeanSquaredError()
  return tf.math.sqrt(mse(y_true, y_pred))

# Taken from https://github.com/ialhashim/DenseDepth/blob/master/Tensorflow/loss.py
def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=1000.0/10.0):
    
    y_pred = tf.expand_dims(y_pred, 3)
    y_true = tf.expand_dims(y_true, 3)

    # Point-wise depth
    l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

    # Edges
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

    # Structural similarity (SSIM) index
    l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)

    # Weights
    w1 = 1.0
    w2 = 1.0
    w3 = theta

    return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))