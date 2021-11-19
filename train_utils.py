
def experiment(name, model, loss, optimizer, batch_size, epochs, augment_rgb=True, augment_spatial=True, shuffle=False, grayscale=False, save_format="h5",
              train=train_ds, test=test_ds):
    model_file = "saved_models/{}.{}".format(name, save_format)
    history_file = "train_history/{}.obj".format(name)
    if os.path.isfile(model_file):
        print("Model file already exists.")
        return
    
    metrics = [losses.percent_relative_error(0.1), losses.percent_relative_error(0.25), 
                   losses.ssim_loss, losses.edge_loss, tf.keras.losses.MeanAbsoluteError(), 
                   tf.keras.losses.MeanSquaredError(), losses.rmse]
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    if augment_spatial:
        train = train.map(spatial_augmentation(out_shape=(224,224)))
    if augment_rgb:
        train = train.map(rgb_augmentation())
    if grayscale:
        train = train.map(to_grayscale)
        test = test.map(to_grayscale)

    train = train.map(resize((224, 224))) 
    if shuffle:
        train = train.shuffle(100, reshuffle_each_iteration=True)
    test = test.map(resize((224, 224)))
    
    train = train.batch(batch_size)
    test = test.batch(16)
    history = model.fit(train,
        epochs=epochs,
#         steps_per_epoch = math.floor(train_size/batch_size),
        verbose = 1,
        validation_data = test)
    model.save(model_file, include_optimizer=False, save_format=save_format)
    
    filehandler = open(history_file,"wb")
    pickle.dump(history.history,filehandler)
    filehandler.close()


def evaluate_model(model, count, grayscale=False, batch_size=1):
    ds = test_ds
    if grayscale:
        ds = ds.map(to_grayscale)
    ds = ds.map(resize((224, 224)))
    ds_batched = ds.batch(batch_size)
    model.compile(metrics=[losses.percent_relative_error(0.1), losses.percent_relative_error(0.25), 
                           losses.ssim_loss, losses.edge_loss, tf.keras.losses.MeanAbsoluteError(), 
                           tf.keras.losses.MeanSquaredError(), losses.rmse])
    model.evaluate(ds_batched)
    
    if count > 0:
        ds = ds.take(count)
        for image, label in ds.as_numpy_iterator():
            f, axarr = plt.subplots(1,3)
            f.set_figheight(5)
            f.set_figwidth(15)
            i = 0

            img = ((image[:, :, 0:3]+1) * 127.5).astype(np.uint8)

            out = model.predict(np.expand_dims(image, axis=0))

            axarr[0].imshow(img)
            axarr[1].imshow(label)
            axarr[2].imshow(out[0, :, :, 0])
            i = i + 1

            plt.show()

def show_history(name):
    file_name = "train_history/{}.obj".format(name)
    if not os.path.isfile(file_name):
        print("No history was saved")
        return
    file = open(file_name,'rb')
    history = pickle.load(file)
    file.close()

    plt.plot(history['loss'], label='Training loss')
    plt.plot(history['val_loss'], label='Validation loss')
    plt.plot(history['relative_error_25.0'], label='Percent under 25% error')
    plt.plot(history['val_relative_error_25.0'], label='Percen undeer 25% error val')
    plt.title('Loss vs Epochs')
    plt.ylabel('MAE value')
    plt.xlabel('Epoch #')
    plt.legend(loc="upper left")
    plt.show()
    
    print(max(history['val_relative_error_25.0']))
    print(history['val_relative_error_25.0'][-1])


def quantized_aware_train(model, loss, optimizer, batch_size, epochs, augment_rgb=True, augment_spatial=True, shuffle=True, grayscale=False):
    quantize_model = tfmot.quantization.keras.quantize_model
    q_model = quantize_model(model)
    metrics = [losses.percent_relative_error(0.1), losses.percent_relative_error(0.25), 
               losses.ssim_loss, losses.edge_loss, tf.keras.losses.MeanAbsoluteError(), 
               tf.keras.losses.MeanSquaredError(), losses.rmse]
    q_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    train = train_ds
    
    if augment_spatial:
        train = train.map(spatial_augmentation(out_shape=(224,224)))
    if augment_rgb:
        train = train.map(rgb_augmentation())
    if grayscale:
        train = train.map(to_grayscale)

    train = train.map(resize((224, 224))) 
    if shuffle:
        train = train.shuffle(100, reshuffle_each_iteration=True)
    
    train = train.batch(batch_size)
    history = q_model.fit(train, epochs=epochs, verbose = 1)
    return q_model
        
        
def save_as_tflite(model, filename, merge_input_channlels=False, quantize=False, grayscale=False):
    def represent_generator():
        ds = test_ds
        if grayscale:
            ds = ds.map(to_grayscale)
        ds = ds.map(resize((224, 224))).take(100).batch(1)
        for image, label in ds.as_numpy_iterator():
            print(image.shape)
            yield [image]
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
#         converter.experimental_new_converter = True
#         converter.experimental_new_quantizer = True
        converter.representative_dataset = tf.lite.RepresentativeDataset(represent_generator)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8
        converter.inference_output_type = tf.uint8  # or tf.uint8
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(filename, 'wb') as f:
        f.write(tflite_model)


def generate_real_samples(image, label, patch_shape):
    y = tf.ones((image.shape[0], patch_shape, patch_shape, 1))
    return [image, label], y

def generate_fake_samples(g_model, image, patch_shape):
    # generate fake instance
    pred_label = g_model.predict(image)
    # create 'fake' class labels (0)
    y = tf.zeros((image.shape[0], patch_shape, patch_shape, 1))
    return pred_label, y

def train_gan(name, generator, discriminator, gan, batch_size, epochs, augment_rgb=True, augment_spatial=True, shuffle=True, grayscale=False):
    generator_file = "saved_models/{}_generator.h5".format(name)
    discriminator_file = "saved_models/{}_discriminator.h5".format(name)
    history_file = "train_history/{}.obj".format(name)
    if os.path.isfile(generator_file):
        print("Model file already exists.")
        return
    
    train = train_ds
    test = test_ds
    if augment_spatial:
        train = train.map(spatial_augmentation(out_shape=(224,224)))
    if augment_rgb:
        train = train.map(rgb_augmentation())
    if grayscale:
        train = train.map(to_grayscale)
        test = test.map(to_grayscale)

    train = train.map(resize((224, 224))) 
    if shuffle:
        train = train.shuffle(100, reshuffle_each_iteration=True)
    test = test.map(resize((224, 224)))
    
    train = train.batch(batch_size, drop_remainder=False)
    test = test.batch(16)
    
    generator.compile(metrics=[losses.percent_relative_error(0.1), losses.percent_relative_error(0.25), 
                       losses.ssim_loss, losses.edge_loss, tf.keras.losses.MeanAbsoluteError(), 
                       tf.keras.losses.MeanSquaredError(), losses.rmse])
    
    for epoch in range(1, epochs+1):
        print("Epoch {}/{}".format(epoch, epochs))
        epoch_start = time.time()
        for i, (image, label) in train.enumerate():
            # select a batch of real samples
            [image, label], y_real = generate_real_samples(image, label, 14)
            # generate a batch of fake samples
            pred_label, y_fake = generate_fake_samples(generator, image, 14)
            # update discriminator for real samples
            d_loss1 = discriminator.train_on_batch([image, label], y_real)
            # update discriminator for generated samples
            d_loss2 = discriminator.train_on_batch([image, pred_label], y_fake)
            # update the generator
            g_loss, _, _ = gan.train_on_batch(image, [y_real, label])
            # summarize performance
            print('>%d, d1[%.3f] d2[%.3f] g[%.3f]\r' % (i+1, d_loss1, d_loss2, g_loss), end="")
        print("")
        generator.evaluate(test)
        epoch_end = time.time()
        print("Epoch took {:.2f} seconds".format(epoch_end-epoch_start))
        
    generator.save(generator_file, include_optimizer=False)
    discriminator.save(discriminator_file, include_optimizer=False)

def spatial_augmentation(seed=1234, out_shape=None):
    rng = tf.random.Generator.from_seed(seed, alg='philox')
    def augmentor(image, label):
        #flip_lr, flip_ud, crop, 
        seed = rng.make_seeds(2)[0]
        label = tf.expand_dims(label, 2)
        image = tf.image.stateless_random_flip_left_right(image, seed)
        label = tf.image.stateless_random_flip_left_right(label, seed)
#         image = tf.image.stateless_random_flip_up_down(image, seed)
#         label = tf.image.stateless_random_flip_up_down(label, seed)
        
#         if rng.uniform((1,))[0] > 0.5:
        scale_factor = rng.uniform((1,), minval=0.5, maxval=1)[0]
        w = image.shape[0]*scale_factor
        h = image.shape[1]*scale_factor
        
        image = tf.image.stateless_random_crop(image, (w, h, 3), seed)
        label = tf.image.stateless_random_crop(label, (w, h, 1), seed)
        
        if out_shape:
            image = tf.image.resize(image, out_shape)
            label = tf.image.resize(label, out_shape)

        image = tf.cast(image, tf.float64)
            
        return image, label[:, :, 0]
    
    return augmentor

def add_zero_channel(image, label):
    t = tf.fill((image.shape[0], image.shape[1], 1), 0.0)
    image = tf.concat([image, t], 2)
    return image, label

def to_grayscale(image, label):
    image = tf.image.rgb_to_grayscale(image)
    return image, label

def rgb_augmentation(seed=1234):
    rng = tf.random.Generator.from_seed(seed, alg='philox')
    def rgb_augmentor(image, label):
        seed = rng.make_seeds(2)[0]
        image = tf.image.stateless_random_brightness(image, 0.05, seed)
        image = tf.image.stateless_random_hue(image, 0.08, seed)
        image = tf.image.stateless_random_saturation(image, 0.6, 1.6, seed)
        image = tf.image.stateless_random_contrast(image, 0.7, 1.3, seed)
        
        return image, label
        
    return rgb_augmentor

def resize(shape):
    def resizer(image, label):
        label = tf.expand_dims(label, 2)
        image = tf.image.resize(image, shape)
        label = tf.image.resize(label, shape)
        
#         image = (image/127.5)-1
#         image = image/255
        
        return image, label[:, :, 0]
        
    return resizer

# spatial_augmentor = spatial_augmentation()
# rgb_augmentor = rgb_augmentation()
# for i in range(30):
#     image, label = next(iter(train_ds))
#     image, label = rgb_augmentor(image, label)
#     image, label = spatial_augmentor(image, label)
#     fig = plt.figure()
#     plt.subplot(1,2,1)
#     plt.title('Original image')
#     plt.imshow(image)
    
#     plt.subplot(1,2,2)
#     plt.title('Augmented image')
#     plt.imshow(label)