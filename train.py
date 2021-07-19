#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

# Copyright (C) 2019 GreenWaves Technologies
# All rights reserved.

# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function

import argparse

import argcomplete
import models
import datasets
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
import numpy as np
import PIL
import PIL.Image

def class_balanced_cross_entropy_loss(label, output):
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss
    """

    labels = tf.cast(tf.greater(label, 0.5), tf.float32)

    num_labels_pos = tf.reduce_sum(labels)
    num_labels_neg = tf.reduce_sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = tf.cast(tf.greater_equal(output, 0), tf.float32)
    loss_val = tf.multiply(output, (labels - output_gt_zero)) - tf.math.log(
        1 + tf.exp(output - 2 * tf.multiply(output, output_gt_zero)))

    loss_pos = tf.reduce_sum(-tf.multiply(labels, loss_val))
    loss_neg = tf.reduce_sum(-tf.multiply(1.0 - labels, loss_val))

    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    return final_loss

def create_parser():
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='train')

    parser.add_argument('h5_file',
                        default="output.h5",
                        nargs=argparse.OPTIONAL,
                        help='Output - Trained model in h5 format')
    parser.add_argument('-b', '--batch_size',
                        type=int,
                        default=128,
                        help='training batch size')
    parser.add_argument('-e', '--epochs',
                        type=int,
                        default=3,
                        help='training epochs')
    parser.add_argument('-B', '--batch_norm',
                        action='store_true',
                        help='carry out batch normalization')
    return parser

def train(args):
    tf.config.list_physical_devices('GPU')
    batch_size = args.batch_size
    num_classes = 10
    epochs = args.epochs

    # input image dimensions
    img_rows, img_cols = 224, 224

    # the data, split between train and test sets
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    (x_train, y_train), (x_test, y_test) = datasets.penutbutter_data()

    # if K.image_data_format() == 'channels_first':
    #     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    #     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    #     input_shape = (1, img_rows, img_cols)
    # else:
    #     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    #     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    #     input_shape = (img_rows, img_cols, 1)

    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')

    # x_train = (x_train / 128) - 1
    # x_test = (x_test / 128) - 1

    # print('x_train shape:', x_train.shape)
    print(x_train.shape, 'train samples')
    print(x_test.shape, 'test samples')

    # convert class vectors to binary class matrices
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)

    model = models.segmentation2(img_rows, img_cols, 1, 10)

    # model.compile(loss=keras.losses.categorical_crossentropy,
    #               optimizer=keras.optimizers.Adam(decay=1e-5),
    #               metrics=['accuracy'])

    learning_rates = [1e-6] #, 1e-7, 1e-8]

    for l in learning_rates:
        model.compile(loss=class_balanced_cross_entropy_loss,
                    optimizer=keras.optimizers.SGD(learning_rate=l, momentum=0.9),
                    metrics=['binary_accuracy'])

        model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_test, y_test))
    # score = model.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
    model.save(args.h5_file)

    for i in range(x_test.shape[0]):
        img = PIL.Image.fromarray((y_test[i, :, :, 0]*255).astype('uint8'), 'L')
        img.save(f'outputs/test-{i}-label.png')
        # img.show()

        out = model.predict(x_test[i:i+1, :, :, :])]
        img = PIL.Image.fromarray((out[0, :, :, 0]*255).astype('uint8'), 'L')
        img.save(f'outputs/test-{i}-output.png')
        # img.show()

    for i in range(x_train.shape[0]):
        img = PIL.Image.fromarray((y_train[i, :, :, 0]*255).astype('uint8'), 'L')
        img.save(f'outputs/train-{i}-label.png')
        # img.show()

        out = model.predict(x_train[i:i+1, :, :, :])
        img = PIL.Image.fromarray((out[0, :, :, 0]*255).astype('uint8'), 'L')
        img.save(f'outputs/train-{i}-output.png')
        # img.show()


def main():
    parser = create_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
