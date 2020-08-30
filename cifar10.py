from pip._internal import main as pipmain
pipmain(['install', 'keras==2.4.3'])
pipmain(['install', 'tensorflow==2.2.0'])

import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import numpy as np

from keras import backend as K
import tensorflow as tf
B = tf.keras.backend

# Load Gelu

@tf.function(experimental_relax_shapes=True)
def gelu(x):
    return 0.5 * x * (1 + B.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))

# Z score for data
# TODO add as an option

def z_score(value, mean, std):
    return (value - mean) / std

# Load Data

(x_train_init, y_train_init), (x_test_init, y_test_init) = cifar10.load_data()

mean = np.mean(x_train_init,axis=(0,1,2,3))
std = np.std(x_train_init,axis=(0,1,2,3))
x_train = z_score(x_train_init, mean, std)
x_test = z_score(x_test_init, mean, std)

num_classes = 10
y_train = np_utils.to_categorical(y_train_init,num_classes)
y_test = np_utils.to_categorical(y_test_init,num_classes)

# Model Architecture

model = Sequential()

# First Section - 32
model.add(Conv2D(32, (3,3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation(gelu))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding='same'))
model.add(Activation(gelu))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
# Second Section - 64
model.add(Conv2D(64, (3,3), padding='same'))
model.add(Activation(gelu))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same'))
model.add(Activation(gelu))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
 
# Third Section - 128
model.add(Conv2D(128, (3,3), padding='same'))
model.add(Activation(gelu))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding='same'))
model.add(Activation(gelu))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.35))
 
# Final dense and Softmax
model.add(Flatten())
model.add(Dense(units=256, activation=gelu))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
 
model.summary()

# LR Helper functions

# Not used yet
def lr_schedule(epoch):
    lrate = init_lr
    if epoch > 9:
        lrate = 0.0025
    if epoch > 19:
        lrate = 0.0015
    if epoch > 29:
        lrate = 0.0075
    if epoch > 44:
        lrate = 0.0035
    return lrate

def lr_decay(epoch):
    return init_lr * decay ** epoch

class LrHistory(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        print("Learning rate:", K.get_value(model.optimizer.lr))
        
early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True),
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='min')

# Hyper Params

batch_size = 64
init_lr = 0.0035
optimizer = keras.optimizers.Adamax()
decay = 0.925
datagen_rate = 4
datagen_rotation = 15

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# TF Datagen

datagen = ImageDataGenerator(
    rotation_range=datagen_rotation,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    )
datagen.fit(x_train)

# Train

model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] * datagen_rate // batch_size,
          epochs=100,
          validation_data=(x_test,y_test),
          callbacks=[
              LearningRateScheduler(lr_decay),
              LrHistory(),
              reduce_lr_loss,
              early_stop
          ])

# Save Model
model.save('cifar')
#testing
scores = model.evaluate(x_test, y_test, batch_size=1)
print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))