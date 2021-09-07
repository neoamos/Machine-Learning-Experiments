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

def mnist():
  num_classes = 10
  img_rows, img_cols = 324, 224
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  if K.image_data_format() == 'channels_first':
      x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
      x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
      input_shape = (1, img_rows, img_cols)
  else:
      x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
      x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
      input_shape = (img_rows, img_cols, 1)

  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')

  x_train = (x_train / 128) - 1
  x_test = (x_test / 128) - 1

  # convert class vectors to binary class matrices
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)

  return (x_train, y_train), (x_test, y_test)

def cocolike_segmentation(path, image_shape=(324,224), mask_shape=(20,14), alphas=[0.3, 0.5, 0.7, 0.9, 1], output_box=True, fake_rgb=False):
  with open(path + 'annotations.json') as json_file:
    data = json.load(json_file)

  annotation_map = {}
  for a in data['annotations']:
    annotation_map[a["image_id"]] = a
  images = []
  masks = []
  bboxes = []
  centerpoint_masks = []
  # random.Random(3).shuffle(data['images'])
  for i in data['images']:
    image = PIL.Image.open(path + i["file_name"]).convert('L')
    # convert image to numpy array
    array = np.asarray(image)
    array = cv2.resize(array, dsize=image_shape, interpolation=cv2.INTER_CUBIC)
    array = (array[:, :, np.newaxis]/127.5)-1
    for a in alphas:
      array_1 = array * a
      if fake_rgb:
        array = np.concatenate([array_1, array_1, array_1], axis=2)
      images.append(array_1)
      images.append(np.rot90(array_1, 2))

      annotation = annotation_map[i['id']]
      bbox = annotation['bbox']

      bboxes.append(np.array([
        (bbox[0]/324)*image_shape[0], 
        (bbox[1]/244)*image_shape[1],
        (bbox[2]/324)*image_shape[0],
        (bbox[3]/244)*image_shape[1],
        ]))
        
      mask = np.zeros((244, 324))
      mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = 1
      mask = cv2.resize(mask, dsize=mask_shape, interpolation=cv2.INTER_CUBIC)
      mask = mask[:, :, np.newaxis]
      masks.append(mask)
      masks.append(np.rot90(mask, 2))

      center_x = round((bbox[0]+(bbox[2]/2))/244*mask_shape[0])
      center_y = round((bbox[1]+(bbox[3]/2))/324*mask_shape[1])

      centerpoint_mask = np.zeros((mask_shape[1], mask_shape[0]))
      # centerpoint_mask[round((bbox[1]+(bbox[3]/2))/244*mask_shape[1])][round((bbox[0]+(bbox[2]/2))/244*mask_shape[1])] = 1
      for x in range(mask_shape[0]):
        for y in range(mask_shape[1]):
          centerpoint_mask[y][x] = abs(center_x-x) + abs(center_y-y)
      # centerpoint_mask = centerpoint_mask.flatten()
      centerpoint_mask = centerpoint_mask[:, :, np.newaxis]
      centerpoint_masks.append(centerpoint_mask)

  images = np.stack(images)
  # images_1 = images*0.8
  # images_2 = images*0.6
  # images_3 = images*0.4
  # images = np.concatenate([images, images_1, images_2, images_3])
  # np.random.shuffle(images)
  masks = np.stack(masks)
  bboxes = np.stack(bboxes)
  centerpoint_masks = np.stack(centerpoint_masks)
  
  train_size = math.floor(len(images) * 0.8)
  test_size = len(images) - train_size

  x_train = images[0:train_size]
  y_train = masks[0:train_size]

  x_test = images[train_size:len(images)]
  y_test = masks[train_size:len(masks)]

  return (x_train, y_train), (x_test, y_test)


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
    depth = depth/10

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