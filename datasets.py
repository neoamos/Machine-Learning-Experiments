import numpy as np
import os
import PIL
import PIL.Image
import json
import tensorflow as tf
import math
import cv2
from keras.datasets import mnist
from keras import backend as K
import random
from PIL import Image, ImageDraw
import h5py
from sklearn.model_selection import train_test_split
import scipy.io
from sklearn.utils import shuffle

def nyu_depth(path, dsize=(320, 224)):
  database = h5py.File(path)

  images = []
  depths = []
  for i in range(database['images'].shape[0]):
    image = database['images'][i]
    image = np.transpose(image)
    image = cv2.resize(image, dsize=dsize)
    image = (image/255)
    images.append(image)

    depth = database['depths'][i]
    depth = np.transpose(depth)
    depth = cv2.resize(depth, dsize=dsize)
    depth = depth/10
    depths.append(depth)

  # images = np.transpose(database['images'], axes=[0, 3, 2, 1])
  # images = images/255
  # depths = np.transpose(database['depths'], axes=[0, 2, 1])
  # depths = depths/10

  images = np.stack(images)
  depths = np.stack(depths)
  return (images, depths)

def nyu_depth_ds(path, train_test_split):
  database = h5py.File(path)
  splits = scipy.io.loadmat('datasets/nyu_depth/splits.mat')
  train_indexes = [i[0] for i in splits['trainNdxs']]
  test_indexes = [i[0] for i in splits['testNdxs']]

  size = database['images'].shape[0]
  # train_size = math.floor((1- train_test_split) * size)
  # test_size = size - train_size

  train_size = len(train_indexes)
  test_size = len(test_indexes)

  def process_example(index):
    image = database['images'][index]
    image = np.transpose(image)
    image = (image/127.5)-1

    depth = database['depths'][index]
    depth = np.transpose(depth)
    depth = (depth/10)

    return image, depth

  def train_generator():
    for i in train_indexes:
      yield process_example(i-1)

  def test_generator():
    for i in test_indexes:
      yield process_example(i-1)

  ds_signature = (
      tf.TensorSpec(shape=(480, 640, 3), dtype=tf.float32),
      tf.TensorSpec(shape=(480, 640), dtype=tf.float32)
    )

  train_ds = tf.data.Dataset.from_generator(train_generator, output_signature=ds_signature).shuffle(train_size).cache(filename="train_cache")
  test_ds = tf.data.Dataset.from_generator(test_generator, output_signature=ds_signature).cache(filename="test_cache")

  return train_ds, test_ds, train_size, test_size


class LargeDataLoader():
  def __init__(self, csv_file='data/nyu2_train.csv', DEBUG=False):
    self.shape_rgb = (480, 640, 3)
    self.shape_depth = (480, 640, 1)
    self.read_nyu_data(csv_file, DEBUG=DEBUG)

  def nyu_resize(self, img, resolution=480, padding=6):
    from skimage.transform import resize
    return resize(img, (resolution, int(resolution*4/3)), preserve_range=True, mode='reflect', anti_aliasing=True )

  def read_nyu_data(self, csv_file, DEBUG=False):
    csv = open(csv_file, 'r').read()
    nyu2_train = list((row.split(',') for row in (csv).split('\n') if len(row) > 0))

    # Dataset shuffling happens here
    nyu2_train = shuffle(nyu2_train, random_state=0)

    # Test on a smaller dataset
    if DEBUG: nyu2_train = nyu2_train[:10]
    
    # A vector of RGB filenames.
    self.filenames = [i[0] for i in nyu2_train]

    # A vector of depth filenames.
    self.labels = [i[1] for i in nyu2_train]

    # Length of dataset
    self.length = len(self.filenames)

  def _parse_function(self, filename, label): 
    # Read images from disk
    image_decoded = tf.image.decode_jpeg(tf.io.read_file(filename))
    depth_resized = tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(label)), [self.shape_depth[0], self.shape_depth[1]])

    # Format
    rgb = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
    depth = tf.image.convert_image_dtype(depth_resized / 255.0, dtype=tf.float32)
    
    # Normalize the depth values (in cm)
    depth = 1000 / tf.clip_by_value(depth * 1000, 10, 1000)

    return rgb, depth


  def get_dataset(self):
    self.dataset = tf.data.Dataset.from_tensor_slices((self.filenames, self.labels))
    self.dataset = self.dataset.shuffle(buffer_size=len(self.filenames), reshuffle_each_iteration=True)
    # self.dataset = self.dataset.repeat()
    self.dataset = self.dataset.map(map_func=self._parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return self.dataset