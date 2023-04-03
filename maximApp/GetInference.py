from ckpt import ckpt, model_handle
import tensorflow as tf
import tensorflow_hub as hub

import matplotlib.pyplot as plt

from PIL import Image
import numpy as np

import sys

from maxim.create_maxim_model import Model
from maxim.maxim.configs import MAXIM_CONFIGS

# Sample Image
image_url = model_handle[1]
image_path = tf.keras.utils.get_file(origin=image_url)
Image.open(image_path)

# Loading Model
z = None
_MODEL = None

# Preprocess
def mod_padding_symmetric(image, factor=64):
    """Padding the image to be divided by factor."""
    height, width = image.shape[0], image.shape[1]
    height_pad, width_pad = ((height + factor) // factor) * factor, (
        (width + factor) // factor
    ) * factor
    padh = height_pad - height if height % factor != 0 else 0
    padw = width_pad - width if width % factor != 0 else 0
    image = tf.pad(
        image, [(padh // 2, padh // 2), (padw // 2, padw // 2), (0, 0)], mode="REFLECT"
    )
    return image


def make_shape_even(image):
    """Pad the image to have even shapes."""
    height, width = image.shape[0], image.shape[1]
    padh = 1 if height % 2 != 0 else 0
    padw = 1 if width % 2 != 0 else 0
    image = tf.pad(image, [(0, padh), (0, padw), (0, 0)], mode="REFLECT")
    return image


def process_image(image: Image):
    input_img = np.asarray(image) / 255.0
    height, width = input_img.shape[0], input_img.shape[1]

    # Padding images to have even shapes
    input_img = make_shape_even(input_img)
    height_even, width_even = input_img.shape[0], input_img.shape[1]

    # padding images to be multiplies of 64
    input_img = mod_padding_symmetric(input_img, factor=64)
    input_img = tf.expand_dims(input_img, axis=0)
    return input_img, height, width, height_even, width_even


def init_new_model(input_img, modelUrl):
    global z, _MODEL
    z = hub.resolve("https://tfhub.dev/sayakpaul/maxim_s-2_deraining_raindrop/1")
    _MODEL = tf.keras.models.load_model(z)
    variant = ckpt.split("/")[-1].split("_")[0]
    # print(variant)
    configs = MAXIM_CONFIGS['S-2']
    # print(configs)
    configs.update(
        {
            "variant": "S-2",
            "dropout_rate": 0.0,
            "num_outputs": 3,
            "use_bias": True,
            "num_supervision_scales": 3,
        }
    )  # From https://github.com/google-research/maxim/blob/main/maxim/run_eval.py#L45-#L61
    configs.update({"input_resolution": (input_img.shape[1], input_img.shape[2])})
    new_model = Model(**configs)
    new_model.set_weights(_MODEL.get_weights())
    return new_model

# Run predictions
def infer(imageArr, modelUrl):
    image = Image.fromarray(imageArr)
    preprocessed_image, height, width, height_even, width_even = process_image(image)
    new_model = init_new_model(preprocessed_image, modelUrl)

    preds = new_model.predict(preprocessed_image)
    if isinstance(preds, list):
        preds = preds[-1]
        if isinstance(preds, list):
            preds = preds[-1]

    preds = np.array(preds[0], np.float32)

    new_height, new_width = preds.shape[0], preds.shape[1]
    h_start = new_height // 2 - height_even // 2
    h_end = h_start + height
    w_start = new_width // 2 - width_even // 2
    w_end = w_start + width
    preds = preds[h_start:h_end, w_start:w_end, :]

    return np.array(np.clip(preds, 0.0, 1.0))

# final_pred_image = infer(image_path)

def imshow(image):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    return image


# plt.figure(figsize=(15, 15))

# plt.subplot(1, 2, 1)
# input_image = np.asarray(Image.open(image_path).convert("RGB"), np.float32) / 255.0
# imshow(input_image, "Input Image")

# plt.subplot(1, 2, 2)
# imshow(final_pred_image, "Predicted Image")

# plt.savefig("./fun.jpg")