import time
import numpy as np
import tensorflow as tf
from metrics import Accuracy_Captha


train_dataset = tf.data.experimental.load('train_dataset')
validation_dataset = tf.data.experimental.load('validation_dataset')
model_1 = tf.keras.models.load_model("assets/models/model_predict_1.h5")
model_2 = tf.keras.models.load_model("assets/models/model_predict_2.h5")


def test_model(model, dataset):
    X_test = []
    y_test = []

    for batch in dataset:
        image, label = batch
        X_test.extend(image)
        y_test.extend(label)

    X_test, y_test = np.array(X_test), np.array(y_test)

    y_pred = model.predict(X_test, verbose=False)

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

start = time.perf_counter()
y_pred = model_1.predict(batch_images, verbose=False)
result_time_1 = (time.perf_counter() - start) * 1000

start = time.time()
y_pred = model_2.predict(batch_images, verbose=False)
result_time_2 = (time.perf_counter() - start) * 1000

with open('result.txt', 'w') as f:
    print('model_1', train_acc_1, valid_acc_1, result_time_1, file=f)
    print('model_2', train_acc_2, valid_acc_2, result_time_2, file=f)
