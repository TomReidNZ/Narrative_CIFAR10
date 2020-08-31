from pip._internal import main as pipmain
pipmain(['install', 'matplotlib'])
pipmain(['install', 'sklearn'])
pipmain(['install', 'seaborn'])

import keras
import numpy as np

from keras.utils import np_utils
from keras.datasets import cifar10

(x_train_init, y_train_init), (x_test_init, y_test_init) = cifar10.load_data()
x_train = (x_train_init).astype('float32')
x_test = (x_test_init).astype('float32')

def z_score(value, mean, std):
    return (value - mean) / std

mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = z_score(x_train, mean, std)
x_test = z_score(x_test, mean, std)

num_classes = 10
y_train = np_utils.to_categorical(y_train_init,num_classes)
y_test = np_utils.to_categorical(y_test_init,num_classes)

reconstructed_model = keras.models.load_model("cifar_model")

scores = reconstructed_model.evaluate(x_test, y_test, batch_size=1, verbose=1)
print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
Y_pred = reconstructed_model.predict(x_test)
y_test_non_category = [ np.argmax(t) for t in y_test ]
y_predict_non_category = [ np.argmax(t) for t in Y_pred ]
conf_mat = confusion_matrix(y_test_non_category, y_predict_non_category)
cnf_matrix_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]


fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='0', cmap=plt.cm.Blues)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix_cifar10.png')

fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cnf_matrix_norm, annot=True, fmt='.2f', cmap=plt.cm.Blues)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix_normalized_cifar10.png')
