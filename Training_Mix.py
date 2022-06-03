from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras import layers, Input, Model
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import pathlib
import Template

current_folder = pathlib.Path(__file__).parent


# Parameters
input_row = 128  # Ch
input_column = 125  # Time
input_color = 1
category_size = 40
batch_size = 128
Epoch = 10000


# Parameter for Sinc Layer
n_filter = 16
filter_dim = 65  # Odd Number
multiplier = 2
sampling_rate = 250
frequency_scale = sampling_rate
min_freq = 1.0
min_band = 4.0
band_initial = 1.0  # Initial Sinc Band: min_band + band_initial
low_freq = 1.0 - min_freq
high_freq = 40.0 - min_freq
seed = 13579


optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)


# Load Data (Visual)
loader = Template.data_block_loader_train_val_test_image_separate(use_all=False, eye_remove=True, do_norm=True, do_zscore=True)

# Load Data (Imagine)
loader_imagine = Template.data_imagination40_loader_train_val_test_sub_all(do_norm=True, do_zscore=True)  # All subject's data
# loader_imagine = Template.data_imagination40_loader_train_val_test_sub001(do_norm=True, do_zscore=True)  # One subject's data
# loader_imagine = Template.data_imagination40_loader_train_val_test_sub002(do_norm=True, do_zscore=True)  # One subject's data
# loader_imagine = Template.data_imagination40_loader_train_val_test_sub003(do_norm=True, do_zscore=True)  # One subject's data
# loader_imagine = Template.data_imagination40_loader_train_val_test_sub004(do_norm=True, do_zscore=True)  # One subject's data

train_data_seen = loader[0]
train_vis_features_seen = loader[1]
train_label_seen = loader[2]
# train_image_seen = loader[3]
validation_data_seen = loader[4]
# validation_vis_features_seen = loader[5]
validation_label_seen = loader[6]
# validation_image_seen = loader[7]
test_dat_seen = loader[8]
# test_vis_features_seen = loader[9]
test_label_seen = loader[10]
counter_seen = loader[11]
seen_flag = np.ones_like(train_label_seen, dtype='float32')  # Flag for calculating MeanSquaredError

train_data_imagine = loader_imagine[0]
train_vis_features_imagine = loader_imagine[1]
train_label_imagine = loader_imagine[2]
# train_image_imagine = loader_imagine[3]
validation_data_imagine = loader_imagine[4]
# validation_vis_features_imagine = loader_imagine[5]
validation_label_imagine = loader_imagine[6]
# validation_image_imagine = loader_imagine[7]
test_dat_imagine = loader_imagine[8]
# test_vis_features_imagine = loader_imagine[9]
test_label_imagine = loader_imagine[10]
counter_imagine = loader_imagine[11]
imagine_flag = np.zeros_like(train_label_imagine, dtype='float32')  # Flag for calculating MeanSquaredError

# Mix data
train_data = np.concatenate([train_data_seen, train_data_imagine], axis=0)
train_features = np.concatenate([train_vis_features_seen, train_vis_features_imagine], axis=0)
train_label = np.concatenate([train_label_seen, train_label_imagine], axis=0)
validation_data = np.concatenate([validation_data_seen, validation_data_imagine], axis=0)
validation_label = np.concatenate([validation_label_seen, validation_label_imagine], axis=0)
test_dat = np.concatenate([test_dat_seen, test_dat_imagine], axis=0)
test_label = np.concatenate([test_label_seen, test_label_imagine], axis=0)
seen_imagine_flag = np.concatenate([seen_flag, imagine_flag], axis=0).reshape((-1, 1))

print('Seen Data')
print(f'Data: {train_data_seen.shape}')  # , Feature: {train_vis_features.shape}')
print(validation_data_seen.shape)
print(test_dat_seen.shape)
print(counter_seen)
print('Imagination Data')
print(f'Data: {train_data_imagine.shape}')  # , Feature: {train_vis_features.shape}')
print(validation_data_imagine.shape)
print(test_dat_imagine.shape)
print(counter_imagine)
print('Total Data')
print(f'Data: {train_data.shape}, Feature: {train_features.shape}')
print(validation_data.shape)
print(test_dat.shape)


def sinc(band, t_right):
    y_right = K.sin(2*np.pi*band*t_right)/(2*np.pi*band*t_right)
    y_left = K.reverse(y_right, 0)
    y = K.concatenate([y_left, [1.0], y_right])
    return y


class SincConv(Layer):

    def __init__(self, n_filter, filter_dim, sr, freq_scale, **kwargs):
        self.n_filter = n_filter
        self.filter_dim = filter_dim
        self.sr = sr
        self.freq_scale = freq_scale

        super(SincConv, self).__init__(**kwargs)

    def get_config(self):
        return {'n_filter': self.n_filter,
                'filter_dim': self.filter_dim,
                'sr': self.sr,
                'freq_scale': self.freq_scale}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        self.filter_b1 = self.add_weight(name='filter_b1',
                                         shape=(self.n_filter,),
                                         initializer='uniform',
                                         trainable=True)
        self.filter_band = self.add_weight(name='filter_band',
                                           shape=(self.n_filter,),
                                           initializer='uniform',
                                           trainable=True)

        np.random.seed(seed)
        initial_b1 = np.random.uniform(low_freq, high_freq, n_filter)
        initial_band = np.zeros_like(initial_b1) + band_initial

        self.set_weights([initial_b1 / self.freq_scale, initial_band / self.freq_scale])

        # Hamming
        n = np.linspace(0, self.filter_dim, self.filter_dim)
        window = 0.54 - 0.46 * K.cos(2 * np.pi * n / self.filter_dim)
        window = K.cast(window, 'float32')
        self.window = K.constant(window, name='window')

        t_right_linspace = np.linspace(1, (self.filter_dim - 1) / 2, int((self.filter_dim - 1) / 2))
        self.t_right = K.constant(t_right_linspace / self.sr, name='t_right')

        super(SincConv, self).build(input_shape)

    def call(self, x, **kwargs):
        self.filter_begin_freq = K.abs(self.filter_b1) + min_freq / self.freq_scale
        self.filter_end_freq = K.clip(self.filter_begin_freq + K.abs(self.filter_band) + min_band / self.freq_scale,
                                      min_freq / self.freq_scale, (self.sr/2) / self.freq_scale)
        filter_list = []
        for i in range(self.n_filter):
            low_pass1 = 2 * self.filter_begin_freq[i] * sinc(self.filter_begin_freq[i] * self.freq_scale, self.t_right)
            low_pass2 = 2 * self.filter_end_freq[i] * sinc(self.filter_end_freq[i] * self.freq_scale, self.t_right)
            band_pass = low_pass2 - low_pass1
            band_pass = band_pass / K.max(band_pass)
            filter_list.append(band_pass * self.window)
        filters = K.stack(filter_list)  # (out_channels, filter_width)
        filters = K.transpose(filters)  # (filter_width, out_channels)
        filters = K.reshape(filters, (1, self.filter_dim, 1, self.n_filter))  # (1, filter_width, 1, out_channels)

        out = K.conv2d(x, kernel=filters, padding='same')

        return out


# Model
def make_model():
    inputs = Input(shape=(input_row, input_column, input_color))
    x = SincConv(n_filter, filter_dim, sampling_rate, frequency_scale)(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.DepthwiseConv2D((input_row, 1), strides=(1, 1), padding='valid', depth_multiplier=multiplier,
                               use_bias=False, depthwise_constraint=tf.keras.constraints.max_norm(1.0))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D((1, 4))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.SeparableConv2D(n_filter*multiplier, (1, 8), strides=(1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D((1, 3))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Flatten()(x)

    features512 = layers.Dense(512, activation='relu', name='features512')(x)
    x = layers.Dropout(0.5)(features512)
    features2048 = layers.Dense(2048, activation='relu', name='features2048')(x)
    x = layers.Dropout(0.5)(features2048)
    outputs = layers.Dense(category_size, activation='softmax', name='outputs',
                           kernel_constraint=tf.keras.constraints.max_norm(0.25))(x)
    model0 = Model(inputs=inputs, outputs=[outputs, features2048])
    return model0


model = make_model()


# Save Model every 100 Epochs
class EpochSave(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % 100 == 0 and epoch+1 != Epoch:
            now0 = datetime.datetime.now()
            now_time0 = now0.strftime('%y%m%d%_H%M%S')
            model_save_name0 = now_time0 + '_Feature_Extractor_Model'
            model.save(current_folder / 'Results' / model_save_name0)


# Tensorboard
log_dir = 'Results/logs/fit/' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# Early Stop
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              mode='min',
                                              min_delta=0.0,
                                              patience=30)


acc = tf.keras.metrics.SparseCategoricalAccuracy()
seen_acc = tf.keras.metrics.SparseCategoricalAccuracy()
imagine_acc = tf.keras.metrics.SparseCategoricalAccuracy()
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
mse = tf.keras.losses.MeanSquaredError()
loss_tracker = tf.keras.metrics.Mean(name='losses')
loss_tracker_feature = tf.keras.metrics.Mean(name='feature_losses')
loss_tracker_total = tf.keras.metrics.Mean(name='total_losses')
loss_tracker_val = tf.keras.metrics.Mean(name='val_losses')
val_acc_history_seen = []
val_loss_history_seen = []
val_acc_history_imagine = []
val_loss_history_imagine = []


# Train Class for Mix Model
class Classifier(Model):
    def __init__(self, base_model):
        super(Classifier, self).__init__()
        self.base_model = base_model

    def compile(self, model_optimizer):
        super(Classifier, self).compile()
        self.model_optimizer = model_optimizer

    def train_step(self, data):
        trains, label = data
        eeg = trains[0]
        vis_features = trains[1]
        flag = trains[2]

        with tf.GradientTape() as tape:
            pred, feature = self.base_model(eeg, training=True)
            loss1 = loss(label, pred)
            loss_feature = mse(flag*vis_features, flag*feature) * tf.cast(tf.size(flag), tf.float32)/(tf.reduce_sum(flag) + 1e-8)
            '''
            Multiply tf.size(flag)/tf.reduce_sum(flag) to normalize
            mse = Error / (batch size * feature size)
            batch size = seen batch size + imagine batch size.
            Change batch size -> seen batch size in mse.
            Add 1e-8 to avoid dividing by 0.
            '''
            total_loss = loss1 + loss_feature
        gradient = tape.gradient(total_loss, self.base_model.trainable_variables)
        self.model_optimizer.apply_gradients(zip(gradient, self.base_model.trainable_variables))

        acc.update_state(label, pred)
        loss_tracker.update_state(loss1)
        loss_tracker_feature.update_state(loss_feature)
        loss_tracker_total.update_state(total_loss)

        return {'acc': acc.result(), 'loss': loss_tracker.result(),
                'feature loss': loss_tracker_feature.result(), 'total': loss_tracker_total.result()}

    def test_step(self, data):
        eeg, label = data
        pred, _ = self.base_model(eeg, training=False)
        seen_acc.update_state(label, pred)
        loss_tracker_val.update_state(loss(label, pred))

        return {'acc': seen_acc.result(), 'loss': loss_tracker_val.result()}

    @property
    def metrics(self):
        return [acc, seen_acc, loss_tracker, loss_tracker_feature, loss_tracker_total, loss_tracker_val]


# Calculate Visual and Imagine Validation Accuracy
class SeparateValidation(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        pred1, _ = model(validation_data_seen, training=False)
        pred2, _ = model(validation_data_imagine, training=False)
        seen_acc.update_state(validation_label_seen, pred1)
        imagine_acc.update_state(validation_label_imagine, pred2)
        val_loss_seen = loss(validation_label_seen, pred1)
        val_loss_imagine = loss(validation_label_imagine, pred2)
        val_acc_history_seen.append(seen_acc.result().numpy())
        val_loss_history_seen.append(val_loss_seen.numpy())
        val_acc_history_imagine.append(imagine_acc.result().numpy())
        val_loss_history_imagine.append(val_loss_imagine.numpy())
        print(f'Visual: Acc = {seen_acc.result().numpy()}, Loss = {val_loss_seen.numpy()}')
        print(f'Imagine: Acc = {imagine_acc.result().numpy()}, Loss = {val_loss_imagine.numpy()}')
        seen_acc.reset_states()
        imagine_acc.reset_states()


# Train
list_of_frequency = []
filter_b1 = model.layers[1].weights[0].numpy() * frequency_scale
filter_band = model.layers[1].weights[1].numpy() * frequency_scale
for k in range(filter_b1.shape[0]):
    filter_begin_freq = np.absolute(filter_b1[k]) + min_freq
    filter_end_freq = filter_begin_freq + np.absolute(filter_band[k]) + min_band
    list_of_frequency.append([filter_begin_freq, filter_end_freq])

second_kernel = model.layers[3].depthwise_kernel.numpy()
second_kernel = np.transpose(second_kernel, (2, 3, 1, 0))
second_kernel = second_kernel.reshape((n_filter*multiplier, 128))


'''
# Plot of initial Sinc layer weights
for i in range(filter_b1.shape[0]):
    plt.plot(list_of_frequency[i], [i+1, i+1], linewidth=0.5)
plt.xlabel('Frequency (Hz)')
plt.ylabel('# of Filter')
plt.show()
for i in range(n_filter):
    plt.subplot(4, 4, i+1)
    plt.plot(second_kernel[2*i], label='1')
    plt.plot(second_kernel[2*i+1], label='2')
    plt.legend()
plt.show()
'''


start = time.time()
classifier = Classifier(base_model=model)
classifier.compile(model_optimizer=optimizer)
history = classifier.fit(x=[train_data, train_features, seen_imagine_flag], y=train_label, epochs=Epoch,
                         batch_size=batch_size,
                         validation_data=(validation_data, validation_label),
                         verbose=2,
                         callbacks=[tensorboard_callback, early_stop, SeparateValidation(), EpochSave()])
elapsed_time = time.time() - start


# Final Test
pred1, _ = model(test_dat_seen, training=False)
pred2, _ = model(test_dat_imagine, training=False)
seen_acc.update_state(test_label_seen, pred1)
imagine_acc.update_state(test_label_imagine, pred2)
test_loss_seen = loss(test_label_seen, pred1)
test_loss_imagine = loss(test_label_imagine, pred2)
print('Test Results')
print('Visual')
print(f'Acc = {seen_acc.result().numpy()}, Loss = {test_loss_seen.numpy()}')
print('Imagine')
print(f'Acc = {imagine_acc.result().numpy()}, Loss = {test_loss_imagine.numpy()}')
seen_acc.reset_states()
imagine_acc.reset_states()
print(f'Time: {elapsed_time} sec')


# Save Final Model and History
now = datetime.datetime.now()
now_time = now.strftime('%y%m%d%_H%M%S')
model_save_name = now_time + '_Feature_Extractor_Model'
model.save(current_folder/'Results'/model_save_name)

history_save_name = 'Results/' + now_time + '_Extractor_History'
np.savez(history_save_name,
         acc=history.history['acc'], loss=history.history['loss'],
         val_acc=history.history['val_acc'], val_loss=history.history['val_loss'],
         feature_loss=history.history['feature loss'], total_loss=history.history['total'],
         val_acc_seen=val_acc_history_seen, val_loss_seen=val_loss_history_seen,
         val_acc_imagine=val_acc_history_imagine, val_loss_imagine=val_loss_history_imagine,
         initial_band=np.array(list_of_frequency),
         initial_depthwise=second_kernel)


# Plot
fig1 = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(history.history['acc'], label='Accuracy')
plt.plot(history.history['val_acc'], label='Val Accuracy')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.subplot(2, 1, 2)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

fig2 = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(val_acc_history_seen, label='Val Acc (Visual)')
plt.plot(val_acc_history_imagine, label='Val Acc (Imagine)')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(val_loss_history_seen, label='Val Loss (Visual)')
plt.plot(val_loss_history_imagine, label='Val Loss (Imagine)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

