import cv2
import numpy as np
import onnxruntime as rt

from fastapi import FastAPI, File

app = FastAPI()

sess = rt.InferenceSession("model.quant.onnx")
input_name = sess.get_inputs()[0].name

max_length = 5
characters = ['2', '3', '4', '5', '6', '7', '8', 'b', 'c', 'd', 'e', 'f', 'g', 'm', 'n', 'p', 'w', 'x', 'y']
img_height, img_width = 200, 50


def get_result(pred):
    """CTC decoder of the output tensor
    https://distill.pub/2017/ctc/
    https://en.wikipedia.org/wiki/Connectionist_temporal_classification
    :return string, float
    """
    last = None
    ans = []
    # pred - 3d tensor, we need 2d array - first element
    for item in pred[0]:
        # get index of element with max accuracy
        char_ind = item.argmax()
        # ignore duplicates and special characters
        if char_ind != last and char_ind != 0 and char_ind != len(characters) + 1:
            # this element is a character - append it to answer
            ans.append(characters[char_ind - 1])
        last = char_ind

    answ = "".join(ans)[:max_length]
    return answ


@app.post("/predict")
def predict(file_bytes: bytes = File()):
    array_data = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    img = cv2.imdecode(array_data, -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255
    img = cv2.resize(img, (img_height, img_width))
    img = cv2.transpose(img)
    img = np.array(img, np.float32)
    img = np.expand_dims(img, axis=(0, -1))

    pred_onx = sess.run(None, {input_name: img})[0]

    answer = get_result(pred_onx)

    return {"predictions": answer}
