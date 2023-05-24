import tensorflow as tf
from config import max_length


max_length = 5
model_dir = "./model"
characters = ['[UNK]', '2', '3', '4', '5', '6', '7', '8', 'b', 'c', 'd', 'e', 'f', 'g', 'm', 'n', 'p', 'w', 'x', 'y']
label2_layer = tf.keras.layers.StringLookup(vocabulary=characters, mask_token=None, invert=True)
model = tf.keras.models.load_model('assets\models\model_1')


class TFModel(tf.Module):
    def __init__(self, model: tf.keras.Model) -> None:
        self.model = model
        self.num_to_char = label2_layer

    def preprocess(self, file_bytes, shape=(50, 200)):
        # 1. Decode and convert to grayscale
        image = tf.reshape(file_bytes, [])
        img = tf.io.decode_png(image, channels=1)
        # 2. Convert to float32 in [0, 1] range
        img = tf.image.convert_image_dtype(img, tf.float32)
        # 4. Resize to the desired size
        img = tf.image.resize(img, shape)
        # 5. Transpose the image because we want the time
        # dimension to correspond to the width of the image.
        img = tf.transpose(img, perm=[1, 0, 2])#perm=[0, 2, 1, 3])
        # 6. Expend dimns img as batch
        img = tf.expand_dims(img, axis=0)
        return img
    
    def postprocess(self, y_pred):
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
        aws = [tf.strings.reduce_join(self.num_to_char(res))]
        return aws

    def f(self):
        @tf.function#(input_signature=[tf.TensorSpec([None], tf.string)])
        def serving_fn(image):
            processed_img = self.preprocess(image)
            probs = self.model(processed_img)
            label = self.postprocess(probs)
            return {"label": tf.expand_dims(label, -1)}
        return serving_fn


model_sig_version = 2
model_sig_export_path = f"{model_dir}/{model_sig_version}"
tf_model_wrapper = TFModel(model)
# trying to create concrete_function as mentioned on github issue
concrete_fn = tf_model_wrapper.f().get_concrete_function(image=tf.TensorSpec([None], tf.string))

model_to_save = tf_model_wrapper.model
model_to_save.num_to_char = tf_model_wrapper.num_to_char


def postprocess(y_pred):
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
    anws = tf.strings.reduce_join(label2_layer(res))
    return anws


def preprocess_bytes(file_bytes, shape=(50, 200)):
    img = tf.reshape(file_bytes, [])
    # 1. Decode and convert to grayscale
    img = tf.io.decode_image(img, channels=1, dtype=tf.uint8, expand_animations=False)
    # 2. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, shape)
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])#perm=[0, 2, 1, 3])
    # 6. Expend dimns img as batch
    img = tf.expand_dims(img, axis=0)
    return img


def preprocess_RGB(image, shape=(50, 200)):
    # 4. Resize to the desired size
    img = tf.image.rgb_to_grayscale(image)
    # 2. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, shape)
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[0, 2, 1, 3])
    # 6. Expend dimns img as batch
    return img


def export_model_1(model):
    @tf.function(input_signature=[tf.TensorSpec(None, tf.string)])
    def serving_fn(image):
        processed_img = preprocess_bytes(image)
        probs = model(processed_img)
        label = postprocess(probs)
        return {"label": tf.expand_dims(label, -1)}

    return serving_fn


def export_model_2(model):
    @tf.function(input_signature=[tf.TensorSpec([None, None, None, 3], tf.uint8)])
    def serving_fn(image):
        processed_img = preprocess_RGB(image)
        probs = model(processed_img)
        label = postprocess(probs)
        return {"label": tf.expand_dims(label, -1)}

    return serving_fn


model_sig_version = 2
model_sig_export_path = f"{model_dir}/{model_sig_version}"

tf.saved_model.save(
    model,
    export_dir=model_sig_export_path,
    signatures={"serving_default": export_model_2(model),
                'serving_bytes': export_model_1(model)},
)