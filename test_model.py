import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend


def ctc_decode(y_pred):
    input_length = tf.ones(y_pred.shape[0], tf.int32) * y_pred.shape[1]
    y_pred = tf.math.log(tf.transpose(y_pred, perm=[1, 0, 2]) + 1e-7)
    (decoded, _) = tf.nn.ctc_greedy_decoder(y_pred, input_length)
    return tf.sparse.to_dense(decoded[0])


def accuracy_captha(y_true, y_pred):
    y_pred = ctc_decode(y_pred)
    y_true = tf.cast(y_true, tf.int64)
    values = tf.math.reduce_all(tf.math.equal(y_true, y_pred), range(1, len(y_true.shape)))
    return tf.cast(values, backend.floatx())


class Accuracy_Captha(keras.metrics.MeanMetricWrapper):
    def __init__(self, name='accuracy_captha', dtype=None):
        super(Accuracy_Captha, self).__init__(
            accuracy_captha, name, dtype=dtype)


train_dataset = tf.data.experimental.load('train_dataset')
validation_dataset = tf.data.experimental.load('validation_dataset')
model_1 = tf.keras.models.load_model("assets/models/model_predict_1.h5")
model_2 = tf.keras.models.load_model("assets/models/model_predict_2.h5")


def test_model(model, dataset):
    X_test = []
    y_test = []

    for batch in dataset:
        X_test.extend(batch["image"])
        y_test.extend(batch["label"])

    X_test, y_test = np.array(X_test), np.array(y_test)

    y_pred = model.predict(X_test)

    m = Accuracy_Captha()
    m.update_state(y_test, y_pred)
    return m.result().numpy()


train_acc_1 = test_model(model_1, train_dataset)
valid_acc_1 = test_model(model_1, validation_dataset)
train_acc_2 = test_model(model_2, train_dataset)
valid_acc_2 = test_model(model_2, validation_dataset)


for batch in validation_dataset.take(1):
    batch_images = batch["image"]
    batch_labels = batch["label"]

start = time.time()
y_pred = model_1.predict(batch_images, verbose=False)
result_time_1 = time.time() - start

start = time.time()
y_pred = model_2.predict(batch_images, verbose=False)
result_time_2 = time.time() - start

with open('result.txt', 'w') as f:
    print('model_1', train_acc_1, valid_acc_1, result_time_1, file=f)
    print('model_2', train_acc_2, valid_acc_2, result_time_2, file=f)
