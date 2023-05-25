import tensorflow as tf
from fastapi import FastAPI, File


app = FastAPI()

model = tf.keras.models.load_model('model.h5')
max_length = 5
characters = ['[UNK]', '2', '3', '4', '5', '6', '7', '8', 'b', 'c', 'd', 'e', 'f', 'g', 'm', 'n', 'p', 'w', 'x', 'y']
img_height, img_width = 50, 200
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=characters, mask_token=None, invert=True
)


def get_result(y_pred):
    input_shape = tf.shape(y_pred)
    num_samples, num_steps = input_shape[0], input_shape[1]
    input_length = tf.ones(num_samples, tf.int32) * num_steps

    y_pred = tf.math.log(tf.transpose(y_pred, perm=[1, 0, 2]) + 1e-7)

    (decoded, _) = tf.nn.ctc_greedy_decoder(y_pred, input_length)

    decoded_dense = []
    for st in decoded:
        st = tf.SparseTensor(st.indices, st.values, (num_samples, num_steps))
        decoded_dense.append(tf.sparse.to_dense(sp_input=st, default_value=-1))

    res = decoded_dense[0][:, :max_length]
    anws = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
    return anws


@app.post("/predict")
def predict(file_bytes: bytes = File()):
    # 1. Decode and convert to grayscale
    img = tf.io.decode_png(file_bytes, channels=1)
    # 2. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # 6. Expend dimns img as batch
    img = tf.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=False)

    answer = get_result(pred)
    return {"predictions": answer}
