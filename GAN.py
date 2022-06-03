from silence_tensorflow import silence_tensorflow
import tensorflow as tf
from tensorflow.keras import layers, Model, models, Input, regularizers, initializers
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import glob
import imageio
import os
import time
import datetime
import pathlib
from PIL import Image
import cv2
import csv
import Template

silence_tensorflow()
tf.config.run_functions_eagerly(True)
# gpu = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu[0], True)


class MoGLayer(layers.Layer):
    def __init__(self,
                 kernel_regularizer=None,
                 kernel_initializer=None,  # 'glorot_uniform',
                 bias_initializer=None,  # 'zeros',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(MoGLayer, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.std = self.add_weight(shape=(input_dim,),
                                   name='std',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer)

        self.mean = self.add_weight(shape=(input_dim,),
                                    initializer=self.bias_initializer,
                                    name='mean')

        self.built = True

    def call(self, inputs, *args, **kwargs):
        output = inputs * self.std
        output = K.bias_add(output, self.mean)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = input_shape[-1]
        return tuple(output_shape)


def time_string(sec_elapsed):
    h = int(sec_elapsed/3600)
    m = int((sec_elapsed % 3600)/60)
    s = sec_elapsed % 60
    return '{}:{:>2}:{:>05.2f}'.format(h, m, s)


# Parameter
image_row = 64
image_column = 64
image_color = 3
batch_size = 16
feature_size = 2048
category_size = 40
noise_size = 2048
Epoch = 1000


# Optimizer
generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)


current_folder = pathlib.Path(__file__).parent


# Models
extractor_file = 'Results/_Feature_Extractor_Model'  # Trained Classifier
extractor = models.load_model(extractor_file, compile=False)
Feature_Layer_Model = Model(inputs=extractor.input, outputs=extractor.get_layer('features2048').output)


# Model for Classifying Generated Images
resnet40_file = 'Data/ResNet_40Class/ResNet_40class_Model'
resnet40 = models.load_model(resnet40_file, compile=False)
resnet40.trainable = False


def make_generator():
    noise_inputs = Input(shape=(noise_size,))
    eeg_inputs = Input(shape=(feature_size,))

    x = MoGLayer(kernel_initializer=initializers.RandomUniform(minval=-0.2, maxval=0.2),
                 bias_initializer=initializers.RandomUniform(minval=-1.0, maxval=1.0),
                 kernel_regularizer=regularizers.l2(0.01))(noise_inputs)
    x = layers.multiply([x, eeg_inputs])

    x = layers.Reshape((1, 1, -1))(x)
    x = layers.Conv2DTranspose(filters=512, kernel_size=4, strides=1, padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2DTranspose(filters=3, kernel_size=4, strides=2, padding='same')(x)
    output = layers.Activation("tanh")(x)

    model = Model(inputs=[noise_inputs, eeg_inputs], outputs=output)
    return model


def make_discriminator():
    image_inputs = Input(shape=(image_row, image_column, image_color))

    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(image_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(256, (1, 1), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(512, (1, 1), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Flatten()(x)

    real_fake = layers.Dense(1024, activation='relu')(x)
    real_fake = layers.Dropout(0.5)(real_fake)
    real_fake = layers.Dense(1, activation='sigmoid')(real_fake)

    category = layers.Dense(1024, activation='relu')(x)
    category = layers.Dropout(0.5)(category)
    category = layers.Dense(1024, activation='relu')(category)
    category = layers.Dropout(0.5)(category)
    category = layers.Dense(category_size, activation='softmax')(category)

    model = Model(inputs=image_inputs, outputs=[real_fake, category])
    return model


generator = make_generator()
discriminator = make_discriminator()


# Create Image Dictionary
data_img_name_label = np.empty((2000, 2), dtype=np.object)
image_dictionary = {}
folder_path = 'Data/Class_Images_Selected'
images_file = 'Data/Class_Images_Selected/Labels.csv'
with open(images_file) as f:
    img_data = csv.reader(f, delimiter='/')
    images_list = [row for row in img_data]
for i in range(len(images_list)):
    img_path = os.path.join(images_list[i][1], images_list[i][2])
    image_name = os.path.join(folder_path, img_path)
    one_image = Image.open(image_name)
    one_image = np.array(one_image.resize((image_row, image_column), Image.LANCZOS))
    if one_image.reshape(-1).shape[0] == image_row*image_column:
        one_image = np.array(Image.open(image_name))
        one_image = cv2.cvtColor(one_image, cv2.COLOR_GRAY2RGB)
        one_image = Image.fromarray(one_image)
        one_image = np.array(one_image.resize((image_row, image_column), Image.LANCZOS))
    one_image = (one_image - 127.5) / 127.5
    one_image = one_image.astype(np.float32)
    image_dictionary[images_list[i][1] + '/' + images_list[i][2]] = one_image
    data_img_name_label[i][0] = images_list[i][1] + '/' + images_list[i][2]
    data_img_name_label[i][1] = int(images_list[i][0])


# Load Data
loader = Template.data_block_loader_train_test_image_separate2(use_all=False, eye_remove=True, do_norm=True, do_zscore=True)

train_eeg = loader[0]
train_labels = loader[1].reshape((-1, 1))
train_images = loader[2].reshape((-1, 1))
test_eeg = loader[3]
test_labels = loader[4]
test_images = loader[5].reshape((-1, 1))
counter = loader[6]

print(train_eeg.shape)
print(test_eeg.shape)
print(counter)


train_features = Feature_Layer_Model.predict(train_eeg)
test_features = Feature_Layer_Model.predict(test_eeg)


base_data = np.concatenate([train_features, train_images, train_labels], axis=1)


# Loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
category_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)


def generator_loss(fake_true_false, fake_pred, label):
    generated_loss = cross_entropy(tf.ones_like(fake_true_false), fake_true_false) + category_loss(label, fake_pred)
    return generated_loss


def discriminator_loss_real(real_true_false, real_pred, label):
    real_loss = cross_entropy(tf.ones_like(real_true_false), real_true_false) + category_loss(label, real_pred)
    return real_loss


def discriminator_loss_fake(fake_true_false):
    fake_loss = cross_entropy(tf.zeros_like(fake_true_false), fake_true_false)
    return fake_loss


loss_tracker_gen = tf.keras.metrics.Mean(name='gen_loss')
loss_tracker_disc = tf.keras.metrics.Mean(name='disc_loss')
loss_tracker_gen_total = tf.keras.metrics.Mean(name='gen_total_loss')
loss_tracker_disc_total = tf.keras.metrics.Mean(name='disc_total_loss')
loss_tracker_test = tf.keras.metrics.Mean(name='test_gen_loss')
acc_gen = tf.keras.metrics.SparseCategoricalAccuracy(name='gen_acc')
acc_disc = tf.keras.metrics.SparseCategoricalAccuracy(name='disc_acc')
acc_test = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc')
rf_d = tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='real')
rf_g = tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='fake')
rf_test = tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='rf')


# GAN Training Class
class GAN(Model):
    def __init__(self, gen_model, disc_model):
        super(GAN, self).__init__()
        self.gen_model = gen_model
        self.disc_model = disc_model

    def compile(self, g_optimizer, d_optimizer):
        super(GAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    def train_step(self, data):
        x, label = data
        feature = x[0]
        real_image_name = x[1].numpy()
        data_size = tf.shape(feature)[0]
        noise = tf.random.normal(shape=(data_size, noise_size))

        real_images = np.empty((data_size, image_row, image_column, image_color), dtype=np.float32)
        for k0 in range(data_size):
            real_images[k0] = image_dictionary[real_image_name[k0][0].decode()]

        # Train Discriminator
        generated_images = self.gen_model([noise, feature], training=True)
        with tf.GradientTape() as tape:
            fake_true_false, _ = self.disc_model(generated_images, training=True)
            real_true_false, real_category = self.disc_model(real_images, training=True)
            disc_loss_real = discriminator_loss_real(real_true_false, real_category, label)
            disc_loss_fake = discriminator_loss_fake(fake_true_false)
            disc_total_loss = disc_loss_real + disc_loss_fake
        gradient = tape.gradient(disc_total_loss, self.disc_model.trainable_variables)
        self.d_optimizer.apply_gradients(zip(gradient, self.disc_model.trainable_variables))

        acc_disc.update_state(label, real_category)
        disc_loss = category_loss(label, real_category)
        rf_d.update_state(tf.ones_like(real_true_false), real_true_false)

        # Train Generator
        with tf.GradientTape() as tape:
            generated_images = self.gen_model([noise, feature], training=True)
            fake_true_false, fake_category = self.disc_model(generated_images, training=True)
            gen_total_loss = generator_loss(fake_true_false, fake_category, label)
        gradient = tape.gradient(gen_total_loss, self.gen_model.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gradient, self.gen_model.trainable_variables))

        rf_g.update_state(tf.ones_like(fake_true_false), fake_true_false)

        resnet_image = tf.image.resize(generated_images, (224, 224), method=tf.image.ResizeMethod.LANCZOS3)
        resnet_image = (resnet_image + 1) / 2 * 255
        resnet_image = tf.keras.applications.resnet.preprocess_input(resnet_image)
        resnet_category = resnet40(resnet_image)
        acc_gen.update_state(label, resnet_category)
        gen_loss = category_loss(label, resnet_category)

        loss_tracker_gen.update_state(gen_loss)
        loss_tracker_disc.update_state(disc_loss)
        loss_tracker_gen_total.update_state(gen_total_loss)
        loss_tracker_disc_total.update_state(disc_total_loss)

        return {'gen_acc': acc_gen.result(), 'gen_loss': loss_tracker_gen.result(),
                'gen_total_loss': loss_tracker_gen_total.result(),
                'disc_acc': acc_disc.result(), 'disc_loss': loss_tracker_disc.result(),
                'disc_total_loss': loss_tracker_disc_total.result(),
                'real': rf_d.result(), 'fake': rf_g.result()}

    def test_step(self, data):
        feature, label = data
        data_size = tf.shape(feature)[0]
        noise = tf.random.normal(shape=(data_size, noise_size))

        generated_image = self.gen_model([noise, feature], training=False)
        fake_real_fake, fake_cat = self.disc_model(generated_image, training=False)
        rf_test.update_state(tf.ones_like(fake_real_fake), fake_real_fake)

        resnet_image = tf.image.resize(generated_image, (224, 224), method=tf.image.ResizeMethod.LANCZOS3)
        resnet_image = (resnet_image + 1) / 2 * 255
        resnet_image = tf.keras.applications.resnet.preprocess_input(resnet_image)
        resnet_cat = resnet40(resnet_image)
        acc_test.update_state(label, resnet_cat)
        gen_loss = category_loss(label, resnet_cat)
        loss_tracker_test.update_state(gen_loss)

        return {'test_acc': acc_test.result(), 'test_gen_loss': loss_tracker_test.result(),
                'rf': rf_test.result()}

    @property
    def metrics(self):
        return [acc_gen, acc_disc, acc_test, loss_tracker_gen, loss_tracker_gen_total,
                loss_tracker_disc, loss_tracker_disc_total, loss_tracker_test, rf_d, rf_g, rf_test]


# Create Generated Images After Each Epoch and Save Model After Each 100 Epoch
class MakeImage(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        generate_and_save_images(self.model.gen_model, epoch+1, seed, sample_feature)
        test_and_save_images(self.model.gen_model, epoch+1)
        if (epoch+1) % 100 == 0 and epoch+1 != Epoch:
            now0 = datetime.datetime.now()
            now_time0 = now0.strftime('%y%m%d%_H%M%S')
            generator_save_name0 = now_time0 + '_Generator_Model'
            discriminator_save_name0 = now_time0 + '_Discriminator_Model'
            generator.save(current_folder / 'Results' / generator_save_name0)
            discriminator.save(current_folder / 'Results' / discriminator_save_name0)


def generate_and_save_images(model, epoch, test_seed, feature):
    # 'training' is set to False
    # This is so all layers run in inference mode (batch norm).
    generated_image = model([test_seed, feature])
    generated_image = (generated_image+1)/2  # 0-1

    fig = plt.figure(figsize=(8, int(np.ceil(category_size/8))))

    for i2 in range(category_size):
        plt.subplot(8, int(np.ceil(category_size/8)), i2+1)
        plt.imshow(generated_image[i2])
        plt.axis('off')

    plt.savefig('Results/Generated_Images/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()
    # plt.show()


def test_and_save_images(model, epoch):
    # 'training' is set to False
    # This is so all layers run in inference mode (batch norm).
    for i3 in range(category_size):
        for ii2 in range(test_labels.shape[0]):
            if test_labels[ii2] == i3:
                # noise = np.random.normal(size=noise_size)
                noise = seed_for_test[i3]
                generated_image = model([np.array([noise]), np.array([test_features[ii2]])])
                generated_image = (generated_image + 1) / 2
                plt.subplot(8, int(np.ceil(category_size/8)), i3+1)
                plt.imshow(generated_image[0])
                plt.axis('off')
                break
    plt.savefig('Results/Tested_Images/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()
    # plt.show()


# Train
start = time.time()
seed = tf.random.normal([category_size, noise_size])
seed_for_test = np.random.normal(size=(category_size, noise_size))
sample_feature = np.empty((category_size, feature_size))  # For making GIF
sample_image = np.empty((category_size, image_row, image_column, image_color))

# Save Real Images in Train Dataset
for k in range(category_size):
    for m in range(len(base_data)):
        if int(base_data[m][-1]) == k:
            sample_feature[k] = base_data[m][:feature_size].astype(np.float32)
            sample_image_name = base_data[m][feature_size:-1]
            sample_img = Image.open('Data/Class_Images_Selected/' + sample_image_name[0])
            sample_img_array = np.asarray(sample_img)
            sample_img_array = Image.fromarray(sample_img_array)
            sample_image[k] = np.array(sample_img_array.resize((image_row, image_column), Image.LANCZOS))
            break
fig0 = plt.figure(figsize=(8, int(np.ceil(category_size / 8))))
for i0 in range(category_size):
    plt.subplot(8, int(np.ceil(category_size / 8)), i0 + 1)
    plt.imshow(sample_image[i0].astype(np.uint8))
    plt.axis('off')
plt.savefig('Results/GAN_real_image.png')
plt.close()


# Save Real Images in Test Dataset
test_sample_image = np.empty((category_size, image_row, image_column, image_color))
for k in range(category_size):
    for m in range(test_labels.shape[0]):
        if test_labels[m] == k:
            test_sample_img = Image.open('Data/Class_Images_Selected/' + test_images[m][0])
            test_sample_img_array = np.asarray(test_sample_img)
            test_sample_img_array = Image.fromarray(test_sample_img_array)
            test_sample_image[k] = np.array(test_sample_img_array.resize((image_row, image_column), Image.LANCZOS))
            break
fig0 = plt.figure(figsize=(8, int(np.ceil(category_size / 8))))
for i0 in range(category_size):
    plt.subplot(8, int(np.ceil(category_size / 8)), i0 + 1)
    plt.imshow(test_sample_image[i0].astype(np.uint8))
    plt.axis('off')
plt.savefig('Results/GAN_real_test_image.png')
plt.close()


gan = GAN(gen_model=generator, disc_model=discriminator)
gan.compile(g_optimizer=generator_optimizer, d_optimizer=discriminator_optimizer)
history = gan.fit(x=[train_features, train_images], y=train_labels, epochs=Epoch, batch_size=batch_size,
                  validation_data=(test_features, test_labels), verbose=2,
                  callbacks=[MakeImage()])
total_time = time.time() - start
print(f'Total Time: {time_string(total_time)}')


# Save Generator and Discriminator
now = datetime.datetime.now()
now_time = now.strftime('%y%m%d%_H%M%S')
generator_save_name = now_time + '_Generator_Model'
discriminator_save_name = now_time + '_Discriminator_Model'
generator.save(current_folder/'Results'/generator_save_name)
discriminator.save(current_folder/'Results'/discriminator_save_name)

save_name = 'Results/' + now_time + '_GAN_Results'
np.savez(save_name,
         gen_accs=history.history['gen_acc'], gen_losses=history.history['gen_loss'],
         gen_total_loss=history.history['gen_total_loss'], disc_total_loss=history.history['disc_total_loss'],
         test_gen_accs=history.history['val_test_acc'], test_gen_losses=history.history['val_test_gen_loss'],
         fake_rf=history.history['fake'], real_rf=history.history['real'], test_rf=history.history['val_rf'])


# Create a GIF
anim_file = now_time + '_GAN_generated_images.gif'
with imageio.get_writer('Results/Generated_Images/' + anim_file, mode='I') as writer:
    filenames = glob.glob('Results/Generated_Images/image*.png')
    filenames = sorted(filenames)
    last = -1
    for ii, filename in enumerate(filenames):
        frame = 2*(ii**0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

# Create a GIF
anim_file2 = now_time + '_GAN_tested_images.gif'
with imageio.get_writer('Results/Tested_Images/' + anim_file2, mode='I') as writer:
    filenames2 = glob.glob('Results/Tested_Images/image*.png')
    filenames2 = sorted(filenames2)
    last = -1
    for ii, filename2 in enumerate(filenames2):
        frame = 2*(ii**0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image2 = imageio.imread(filename2)
        writer.append_data(image2)
    image2 = imageio.imread(filename2)
    writer.append_data(image2)


# Plot
fig1 = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(history.history['gen_acc'], label='Generator Accuracy')
plt.plot([0, Epoch], [0.025, 0.025], color='gray', linestyle='dashed')
plt.xlabel('Epoch')
plt.ylabel('Train Accuracy')
plt.legend(loc='lower right')
plt.subplot(2, 1, 2)
plt.plot(history.history['gen_loss'], label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.legend(loc='upper right')

fig2 = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(history.history['disc_acc'], label='Discriminator Accuracy')
plt.plot([0, Epoch], [0.025, 0.025], color='gray', linestyle='dashed')
plt.xlabel('Epoch')
plt.ylabel('Train Accuracy')
plt.legend(loc='lower right')
plt.subplot(2, 1, 2)
plt.plot(history.history['disc_loss'], label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.legend(loc='upper right')

fig3 = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(history.history['gen_total_loss'], label='Generator Total Loss')
plt.ylabel('Total Loss')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(history.history['disc_total_loss'], label='Discriminator Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.legend()

fig4 = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(history.history['val_test_acc'], label='Generator Test Accuracy')
plt.plot([0, Epoch], [0.025, 0.025], color='gray', linestyle='dashed')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.legend(loc='lower right')
plt.subplot(2, 1, 2)
plt.plot(history.history['val_test_gen_loss'], label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Test Loss')
plt.legend(loc='upper left')

fig5 = plt.figure()
plt.plot(history.history['fake'], label='Fake True False')
plt.plot(history.history['real'], label='Real True False')
plt.xlabel('Epoch')
plt.legend()

fig6 = plt.figure()
plt.plot(history.history['val_rf'], label='Test True False')
plt.xlabel('Epoch')
plt.legend()


plt.show()

