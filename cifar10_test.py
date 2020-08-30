import keras
import numpy as np

from keras.utils import np_utils
from keras.datasets import cifar10

(x_train_init, y_train), (x_test_init, y_test) = cifar10.load_data()

def z_score(value, mean, std):
    return (value - mean) / std

mean = np.mean(x_train_init,axis=(0,1,2,3))
std = np.std(x_train_init,axis=(0,1,2,3))
x_train = z_score(x_train_init, mean, std)
x_test = z_score(x_test_init, mean, std)

num_classes = 10
y_train = np_utils.to_categorical(y_train,num_classes)
y_test = np_utils.to_categorical(y_test,num_classes)

reconstructed_model = keras.models.load_model("model")

scores = reconstructed_model.evaluate(x_test, y_test, batch_size=1, verbose=1)
print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))