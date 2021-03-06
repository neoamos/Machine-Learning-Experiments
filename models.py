
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, UpSampling2D, LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Add, SeparableConv2D
from tensorflow.keras.models import Sequential
from tensorflow_addons.layers import SpectralNormalization
from tensorflow.keras import regularizers

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

from tensorflow.keras.applications import MobileNetV2, DenseNet169


def mnist_cnn(img_width, img_height, img_channels, output_dim, batch_norm=False):
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3), strides=(2, 2), input_shape=(img_width, img_height, 1)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3), strides=(1, 1)))
	if batch_norm:
			model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(output_dim))
	if batch_norm:
			model.add(BatchNormalization())
	model.add(Activation('softmax'))
	optimizer = keras.optimizers.Adam(decay=1e-5)
	loss = keras.losses.categorical_crossentropy
	metrics = ['accuracy']

	print(model.summary())
	return (model, optimizer, loss, metrics)

def vgg_segmentation(img_width, img_height, img_channels, upsample=False):
	batch_norm = True
	img_input = Input(shape=(img_height, img_width, img_channels))

	x1 = Conv2D(16, (5, 5), strides=(2,2), padding='same', kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(1e-4))(img_input)
	if batch_norm:
		x1 = BatchNormalization()(x1)
	x1 = Activation('relu')(x1)
	x1 = Conv2D(16, (3, 3), padding='same', kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(1e-4))(x1)
	if batch_norm:
		x1 = BatchNormalization()(x1)
	x1 = Activation('relu')(x1)

	x2 = MaxPooling2D(strides=[2,2])(x1)

	x2 = Conv2D(16, (3, 3), padding='same', kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(1e-4))(x2)
	if batch_norm:
		x2 = BatchNormalization()(x2)
	x2 = Activation('relu')(x2)
	x2 = Conv2D(16, (3, 3), padding='same', kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(1e-4))(x2)
	if batch_norm:
		x2 = BatchNormalization()(x2)
	x2 = Activation('relu')(x2)

	x3 = MaxPooling2D(strides=[2,2])(x2)

	x3 = Conv2D(32, (3, 3), padding='same', kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(1e-4))(x3)
	if batch_norm:
		x3 = BatchNormalization()(x3)
	x3 = Activation('relu')(x3)
	x3 = Conv2D(32, (3, 3), padding='same', kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(1e-4))(x3)
	if batch_norm:
		x3 = BatchNormalization()(x3)
	x3 = Activation('relu')(x3)
	x3 = Conv2D(32, (3, 3), padding='same', kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(1e-4))(x3)
	if batch_norm:
		x3 = BatchNormalization()(x3)
	x3 = Activation('relu')(x3)

	x4 = MaxPooling2D(strides=[2,2])(x3)

	x4 = Conv2D(32, (3, 3), padding='same', kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(1e-4))(x4)
	if batch_norm:
		x4 = BatchNormalization()(x4)
	x4 = Activation('relu')(x4)
	x4 = Conv2D(32, (3, 3), padding='same', kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(1e-4))(x4)
	if batch_norm:
		x4 = BatchNormalization()(x4)
	x4 = Activation('relu')(x4)
	x4 = Conv2D(32, (3, 3), padding='same', kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(1e-4))(x4)
	if batch_norm:
		x4 = BatchNormalization()(x4)
	x4 = Activation('relu')(x4)

	x5 = MaxPooling2D(strides=[2,2])(x4)

	x5 = Conv2D(32, (3, 3), padding='same', kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(1e-4))(x5)
	if batch_norm:
		x5 = BatchNormalization()(x5)
	x5 = Activation('relu')(x5)
	x5 = Conv2D(32, (3, 3), padding='same', kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(1e-4))(x5)
	if batch_norm:
		x5 = BatchNormalization()(x5)
	x5 = Activation('relu')(x5)
	x5 = Conv2D(32, (3, 3), padding='same', kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(1e-4))(x5)
	if batch_norm:
		x5 = BatchNormalization()(x5)
	x5 = Activation('relu')(x5)

	# Skip connections can be used to increase resolution of the output
	# But it also requires a lot of intermediate values to be stored
	if upsample:
		x1 = UpSampling2D(size=(2,2))(x1)
		x2 = UpSampling2D(size=(4,4))(x2)
		x3 = UpSampling2D(size=(8,8))(x3)
		x4 = UpSampling2D(size=(16,16))(x4)
		x5 = UpSampling2D(size=(32,32))(x5)
		x = concatenate([x1, x2, x3, x4, x5])
	else:
		x = x4

	# x = Conv2D(10, (3, 3), padding='same', kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(1e-4))(x)
	# x = BatchNormalization()(x)
	# x = Activation('sigmoid')(x)
	# x = Conv2D(5, (3, 3), padding='same', kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(1e-4))(x)
	# x = BatchNormalization()(x)
	# x = Activation('sigmoid')(x)
	x = Conv2D(1, (1, 1), padding='same', kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(1e-4))(x)
	x = Activation('sigmoid')(x)

	model = Model(inputs=[img_input], outputs=[x])
	optimizer = keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9)
	# optimizer = keras.optimizers.Adam(learning_rate=0.006, decay=1e-5)
	loss = class_balanced_cross_entropy_loss
	# loss = keras.losses.categorical_crossentropy
	metrics = ['accuracy']
	print(model.summary())
	return (model, optimizer, loss, metrics)

def mobilenet_segmentation(img_width, img_height, img_channels):
	base_model = keras.applications.MobileNetV2(
		input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet', alpha=0.5)
	
	base_model.trainable = False
	x = base_model.get_layer('block_13_expand_relu').output
	# base_model.get_layer('block_13_expand_relu').traibale = False

	# img_input = keras.Input(shape=(img_height, img_width, 3))
	# x = base_model(img_input)
	x = Conv2D(1, (1, 1), padding='same', kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(1e-4))(x)
	mask = Activation('relu')(x)

	model = Model(inputs=[base_model.input], outputs=[mask])
	# for l in model.layers[0:-5]:
	#    print(l)
	#    l.trainable = False
	optimizer = keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9)
	loss = class_balanced_cross_entropy_loss
	metrics = ['binary_accuracy']
	# print(model.summary())
	return (model, optimizer, loss, metrics)


def resnet8_segmentation(img_width, img_height, img_channels):
	"""
	Define model architecture.
	
	# Arguments
		img_width: Target image widht.
		img_height: Target image height.
		img_channels: Target image channels.
		output_dim: Dimension of model output.
		
	# Returns
		model: A Model instance.
	"""

	# Input
	img_input = Input(shape=(img_height, img_width, img_channels))

	x1 = Conv2D(32, (5, 5), strides=[2,2], padding='same')(img_input)
	x1 = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(x1)

	# First residual block
	x2 = BatchNormalization()(x1)
	x2 = Activation('relu')(x2)
	x2 = Conv2D(32, (3, 3), strides=[2,2], padding='same',
							kernel_initializer="he_normal",
							kernel_regularizer=regularizers.l2(1e-4))(x2)

	x2 = BatchNormalization()(x2)
	x2 = Activation('relu')(x2)
	x2 = Conv2D(32, (3, 3), padding='same',
							kernel_initializer="he_normal",
							kernel_regularizer=regularizers.l2(1e-4))(x2)

	x1 = Conv2D(32, (1, 1), strides=[2,2], padding='same')(x1)
	x3 = add([x1, x2])

	# Second residual block
	x4 = BatchNormalization()(x3)
	x4 = Activation('relu')(x4)
	x4 = Conv2D(64, (3, 3), strides=[2,2], padding='same',
							kernel_initializer="he_normal",
							kernel_regularizer=regularizers.l2(1e-4))(x4)

	x4 = BatchNormalization()(x4)
	x4 = Activation('relu')(x4)
	x4 = Conv2D(64, (3, 3), padding='same',
							kernel_initializer="he_normal",
							kernel_regularizer=regularizers.l2(1e-4))(x4)

	x3 = Conv2D(64, (1, 1), strides=[2,2], padding='same')(x3)
	x5 = add([x3, x4])

	# Third residual block
	# x6 = BatchNormalization()(x5)
	# x6 = Activation('relu')(x6)
	# x6 = Conv2D(128, (3, 3), strides=[2,2], padding='same',
	#             kernel_initializer="he_normal",
	#             kernel_regularizer=regularizers.l2(1e-4))(x6)

	# x6 = BatchNormalization()(x6)
	# x6 = Activation('relu')(x6)
	# x6 = Conv2D(128, (3, 3), padding='same',
	#             kernel_initializer="he_normal",
	#             kernel_regularizer=regularizers.l2(1e-4))(x6)

	# x5 = Conv2D(128, (1, 1), strides=[2,2], padding='same')(x5)
	# x7 = add([x5, x6])

	x = Activation('sigmoid')(x5)

	# Collision channel
	# coll = Dense(output_dim)(x)
	# coll = Activation('sigmoid')(coll)

	# Define steering-collision model
	model = Model(inputs=[img_input], outputs=[x])
	print(model.summary())

	optimizer = keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9)
	optimizer = keras.optimizers.Adam(learning_rate=0.006, decay=1e-5)
	loss = class_balanced_cross_entropy_loss
	metrics = ['accuracy']
	print(model.summary())
	return (model, optimizer, loss, metrics)

def resnet8(img_width, img_height, img_channels, output_dim):
	"""
	Define model architecture.
	
	# Arguments
		img_width: Target image widht.
		img_height: Target image height.
		img_channels: Target image channels.
		output_dim: Dimension of model output.
		
	# Returns
		model: A Model instance.
	"""

	# Input
	img_input = Input(shape=(img_height, img_width, img_channels))

	x1 = Conv2D(32, (5, 5), strides=[2,2], padding='same')(img_input)
	x1 = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(x1)

	# First residual block
	x2 = BatchNormalization()(x1)
	x2 = Activation('relu')(x2)
	x2 = Conv2D(32, (3, 3), strides=[2,2], padding='same',
							kernel_initializer="he_normal",
							kernel_regularizer=regularizers.l2(1e-4))(x2)

	x2 = BatchNormalization()(x2)
	x2 = Activation('relu')(x2)
	x2 = Conv2D(32, (3, 3), padding='same',
							kernel_initializer="he_normal",
							kernel_regularizer=regularizers.l2(1e-4))(x2)

	x1 = Conv2D(32, (1, 1), strides=[2,2], padding='same')(x1)
	x3 = add([x1, x2])

	# Second residual block
	x4 = BatchNormalization()(x3)
	x4 = Activation('relu')(x4)
	x4 = Conv2D(64, (3, 3), strides=[2,2], padding='same',
							kernel_initializer="he_normal",
							kernel_regularizer=regularizers.l2(1e-4))(x4)

	x4 = BatchNormalization()(x4)
	x4 = Activation('relu')(x4)
	x4 = Conv2D(64, (3, 3), padding='same',
							kernel_initializer="he_normal",
							kernel_regularizer=regularizers.l2(1e-4))(x4)

	x3 = Conv2D(64, (1, 1), strides=[2,2], padding='same')(x3)
	x5 = add([x3, x4])

	# Third residual block
	x6 = BatchNormalization()(x5)
	x6 = Activation('relu')(x6)
	x6 = Conv2D(128, (3, 3), strides=[2,2], padding='same',
							kernel_initializer="he_normal",
							kernel_regularizer=regularizers.l2(1e-4))(x6)

	x6 = BatchNormalization()(x6)
	x6 = Activation('relu')(x6)
	x6 = Conv2D(128, (3, 3), padding='same',
							kernel_initializer="he_normal",
							kernel_regularizer=regularizers.l2(1e-4))(x6)

	x5 = Conv2D(128, (1, 1), strides=[2,2], padding='same')(x5)
	x7 = add([x5, x6])

	x = Flatten()(x7)
	x = Activation('relu')(x)
	x = Dropout(0.5)(x)

	# Steering channel
	x = Dense(output_dim)(x)
	nums = Activation('softmax')(x)

	# Collision channel
	# coll = Dense(output_dim)(x)
	# coll = Activation('sigmoid')(coll)

	# Define steering-collision model
	model = Model(inputs=[img_input], outputs=[nums])
	print(model.summary())

	return model


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

	return final_loss*900

def unet(img_width, img_height, img_channels):
	input_size = (img_width, img_height, img_channels)
	inputs = Input(input_size)
	conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
	conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
	conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
	conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
	conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
	conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
	drop4 = Dropout(0.5)(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

	conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
	conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
	drop5 = Dropout(0.5)(conv5)

	up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
	merge6 = concatenate([drop4,up6], axis = 3)
	conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
	conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

	up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
	merge7 = concatenate([conv3,up7], axis = 3)
	conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
	conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

	up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
	merge8 = concatenate([conv2,up8], axis = 3)
	conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
	conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

	up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
	merge9 = concatenate([conv1,up9], axis = 3)
	conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
	conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
	conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
	conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

	model = Model(inputs=[inputs], outputs=[conv10])

	# model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss = keras.losses.binary_crossentropy, metrics = ['accuracy'])

	# model.compile(loss=keras.losses.binary_crossentropy,
	#               optimizer=keras.optimizers.Adam(learning_rate=1e-4),
	#               metrics=['accuracy'])
	print(model.summary())

	return model

# def upconv(x, out_chan, skip, conv_size):
# 	x = UpSampling2D((2, 2))(x)
# 	if skip != None:
# 		x = Concatenate()([x, skip])
	
# 	x = Conv2D(out_chan, (conv_size, conv_size), padding="same", kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4))(x)
# 	x = BatchNormalization()(x)
# 	x = Activation("relu")(x)

# 	return x

def upconv(x, out_chan, skip, conv_size, separable_conv=False, relu6=False, squeeze=True, spectral_normalization=False):
	normalize = SpectralNormalization if spectral_normalization else lambda x: x
	conv = SeparableConv2D if separable_conv else Conv2D
	relu = tf.keras.layers.ReLU(6.0) if relu6 else Activation("relu")
	
	if squeeze:
		x = normalize(Conv2D(x.shape[-1]/6, (1, 1), padding="same", kernel_initializer='he_normal'))(x)
	x = UpSampling2D((2, 2))(x)
	if skip != None:
		bottleneck = skip.shape[-1]/6
		if bottleneck < 3:
			bottleneck = 3
		if squeeze:
			skip = normalize(Conv2D(bottleneck, (1, 1), padding="same", kernel_initializer='he_normal'))(skip)
		x = Concatenate()([x, skip])

	x = normalize(conv(out_chan, (conv_size, conv_size), padding="same", kernel_initializer='he_normal'))(x)
	x = BatchNormalization()(x)
	x = relu(x)

	return x

def mobilenet_unet(img_width, img_height, img_channels, weights="imagenet", 
	alpha=0.35, regularize_decoder=True, skip_connections=True, decoder_initializer='he_normal',
	upconv_size=3, skip_input=True, separable_conv=False, relu6=False, squeeze=True, spectral_normalization=False):
	if regularize_decoder:
		regularizer = regularizers.l2(1e-4)
	else:
		regularizer = None

	inputs = Input(shape=(img_height, img_width, img_channels), name="input_image")

	if weights == "imagenet" and img_channels > 3:
		# Use the same weights for first three channels, but new weights for new channels
		inputs_3 = Input(shape=(img_height, img_width, 3), name="input_image")
		encoder_pt = MobileNetV2(input_tensor=inputs_3, weights=weights, include_top=False, alpha=alpha)
		encoder = MobileNetV2(input_tensor=inputs, weights=None, include_top=False, alpha=alpha)
		for layer_pt in encoder_pt.layers:
				layer = encoder.get_layer(layer_pt.name)
				if layer_pt.name == "Conv1":
						pt_weights = layer_pt.get_weights()[0]
						weights = layer.get_weights()[0]
						new_weights = tf.concat([pt_weights, weights[:, :, 3:, :]], 2)
						layer.set_weights([new_weights])
				else:
						layer.set_weights(layer_pt.get_weights())
	elif weights == "imagenet" and img_channels == 1:
		# Fuse layer 1 weights into a single filter channel
		# Essentially the same as having 3 input channels with identical values
		inputs_3 = Input(shape=(img_height, img_width, 3), name="input_image")
		encoder_pt = MobileNetV2(input_tensor=inputs_3, weights="imagenet", include_top=False, alpha=alpha)
		encoder = MobileNetV2(input_tensor=inputs, weights=None, include_top=False, alpha=alpha)
		for layer_pt in encoder_pt.layers:
			layer = encoder.get_layer(layer_pt.name)
			if layer_pt.name == "Conv1":
					pt_weights = layer_pt.get_weights()[0]
					new_weights = tf.reduce_sum(pt_weights, axis=2, keepdims=True)
					layer.set_weights([new_weights])
			else:
					layer.set_weights(layer_pt.get_weights())
	else:
		encoder = MobileNetV2(input_tensor=inputs, weights=weights, include_top=False, alpha=alpha)

	if skip_input:
		skip_input = "input_image"
	else:
		skip_input = None
	upconv_layers = range(5)
	skip_connection_names = ["block_13_expand_relu", "block_6_expand_relu", "block_3_expand_relu", "block_1_expand_relu", skip_input]

	normalize = SpectralNormalization if spectral_normalization else lambda x: x
	encoder_output = encoder.get_layer("block_16_expand_relu").output
	x = encoder_output
	for i in upconv_layers:
		if skip_connections and skip_connection_names[i] != None:
			skip = encoder.get_layer(skip_connection_names[i]).output
		else:
			skip = None

		x = upconv(x, x.shape[-1]/2, skip, upconv_size, separable_conv=separable_conv, relu6=relu6, squeeze=squeeze, spectral_normalization=spectral_normalization)
		
	x = normalize(Conv2D(1, (3, 3), padding="same", kernel_initializer=decoder_initializer, kernel_regularizer=regularizer))(x)
	x = Activation("sigmoid")(x)

	model = Model(inputs, x)

	# print(model.summary())
	return model


def discriminator(image_shape, spectral_normalization=False):
	# weight initialization
	init = 'he_normal'
	# source image input
	in_src_image = Input(shape=image_shape)
	# target image input
	in_target_image = Input(shape=(224,224,1))
	# concatenate images channel-wise
	normalize = SpectralNormalization if spectral_normalization else lambda x: x

	merged = Concatenate()([in_src_image, in_target_image])
	# C64
	d = normalize(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init))(merged)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = normalize(Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = normalize(Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init))(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = normalize(Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init))(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
	d = normalize(Conv2D(512, (4,4), padding='same', kernel_initializer=init))(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	d = normalize(Conv2D(1, (4,4), padding='same', kernel_initializer=init))(d)
	patch_out = Activation('sigmoid')(d)
	# define model
	model = Model([in_src_image, in_target_image], patch_out)
	return model



# define the combined generator and discriminator model, for updating the generator
def gan(g_model, d_model, image_shape):
	# make weights in the discriminator not trainable
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	# define the source image
	in_src = Input(shape=image_shape)
	# connect the source image to the generator input
	gen_out = g_model(in_src)
	# connect the source input and generator output to the discriminator input
	dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and classification output
	model = Model(in_src, [dis_out, gen_out])
	# compile model
	return model



def dense_depth_upscale_block(filters, name, x, skip):      
	x = UpSampling2D(size=(2, 2), interpolation='bilinear', name=name+'_upsampling2d')(x)
	x = Concatenate(name=name+'_concat')([x, skip]) # Skip connection        
	x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convA')(x)
	x = LeakyReLU(alpha=0.2)(x)
	x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convB')(x)
	x = LeakyReLU(alpha=0.2)(x)
	return x

def dense_depth_encoder(input):               
	base_model = DenseNet169(input_tensor=input, include_top=False, weights='imagenet')   
	print('Base model loaded {}'.format(DenseNet169.__name__))
	
	# Create encoder model that produce final features along with multiple intermediate features
	features = [base_model.outputs[-1]]
	for name in ['pool1', 'pool2_pool', 'pool3_pool', 'conv1/relu'] : features.append( base_model.get_layer(name).output )        
	return features
	
def dense_depth_decoder(decode_filters, features):
	x, pool1, pool2, pool3, conv1 = features[0], features[1], features[2], features[3], features[4]       
	x =  Conv2D(filters=decode_filters, kernel_size=1, padding='same', name='conv2')(x)     
	x = dense_depth_upscale_block(decode_filters//2,  'up1', x, pool3)
	x = dense_depth_upscale_block(decode_filters//4,  'up2', x, pool2)
	x = dense_depth_upscale_block(decode_filters//8,  'up3', x, pool1)
	x = dense_depth_upscale_block(decode_filters//16, 'up4', x, conv1)
	x = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv3')(x)
	return x
    
def dense_depth(image_shape):
	input = Input(shape=image_shape)
	features = dense_depth_encoder(input)
	out = dense_depth_decoder( int(features[0].shape[-1] // 2 ), features )
	out = UpSampling2D((2, 2))(out)

	model = Model(input, out)
	return model