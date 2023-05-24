import os
import cv2
import numpy as np
from onnxruntime.quantization import quantize_static, CalibrationDataReader


def preprocess_image(image_filepath):
    img = cv2.imread(os.path.join(image_filepath))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255
    img = cv2.resize(img, (200, 50))
    img = cv2.transpose(img)
    img = np.array(img, np.float32)
    return img


def preprocess_func(images_folder, size_limit=0):
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = images_folder + '/' + image_name
        image_data = preprocess_image(image_filepath)
        unconcatenated_batch_data.append(image_data)
    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=(0, -1)), axis=0)
    print(batch_data.shape)
    return batch_data


class MobilenetDataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder):
        self.image_folder = calibration_image_folder
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            nhwc_data_list = preprocess_func(self.image_folder, size_limit=0)
            self.datasize = len(nhwc_data_list)
            self.enum_data_dicts = iter([{'image': np.expand_dims(nhwc_data, 0)} for nhwc_data in nhwc_data_list])
        return next(self.enum_data_dicts, None)


model_fp32 = 'deploy_onnx/model.onnx'
model_quant = 'deploy_onnx/model.quant.onnx'
dr = MobilenetDataReader('captcha_images_v2')
quantize_static(model_fp32, model_quant, dr)
