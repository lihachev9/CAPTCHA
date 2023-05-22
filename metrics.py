import tensorflow as tf
from config import max_length


def ctc_decode(y_pred):
    input_shape = tf.shape(y_pred)
    num_samples, num_steps = input_shape[0], input_shape[1]
    input_length = tf.ones(num_samples, tf.int32) * num_steps

    y_pred = tf.math.log(tf.transpose(y_pred, perm=[1, 0, 2]) + 1e-7)

    (decoded, _) = tf.nn.ctc_greedy_decoder(y_pred, input_length)

    decoded_dense = []
    for st in decoded:
        st = tf.SparseTensor(st.indices, st.values, (num_samples, num_steps))
        decoded_dense.append(tf.sparse.to_dense(sp_input=st, default_value=-1))

    return decoded_dense[0][:, :max_length]


def accuracy_captha(y_true, y_pred):
    y_pred = ctc_decode(y_pred)
    y_true = tf.cast(y_true, tf.int64)
    values = tf.math.reduce_all(tf.math.equal(y_true, y_pred), range(1, len(y_true.shape)))
    values = False == values
    values = tf.cast(values, tf.float32)
    return values


class WER(tf.keras.metrics.MeanMetricWrapper):
    def __init__(self, name='WER', dtype=None):
        super(WER, self).__init__(
            accuracy_captha, name, dtype=dtype)
