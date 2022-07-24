import numpy as np
from PIL import Image
import tensorflow as tf

tf.config.experimental.set_visible_devices([], "GPU")
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logging.info("Starting tensorfood...")
from typing import Tuple

FOODS = [
    "chilli_crab",
    "curry_puff",
    "dim_sum",
    "ice_kacang",
    "kaya_toast",
    "nasi_ayam",
    "popiah",
    "roti_prata",
    "sambal_stingray",
    "satay",
    "tau_huay",
    "wanton_noodle",
]

PRED_MAP = {
    0: "chilli_crab",
    1: "curry_puff",
    2: "dim_sum",
    3: "ice_kacang",
    4: "kaya_toast",
    5: "nasi_ayam",
    6: "popiah",
    7: "roti_prata",
    8: "sambal_stingray",
    9: "satay",
    10: "tau_huay",
    11: "wanton_noodle",
}

# FOOD_PATH = "./images/kt.jpg"
# to run locally, uncomment below -
MODEL_PATH = "tensorfood.h5"


def get_model(model_path=MODEL_PATH) -> tf.keras.Model:
    """To load in the trained model (tensorfood.h5) for prediction

    Args:
        model_path (optional): Defaults to MODEL_PATH.

    Returns:
        tf.keras.Model: A keras model instance
    """
    logging.info("Loading model...")
    return tf.keras.models.load_model(model_path)


def get_prediction(image_path: str) -> Tuple[str, str]:
    """Accept an image via path, and return two things
    1. predicted food label in string
    2. proba of predicted food class

    Args:
        image_path (str): Path to test image

    Returns:
        Tuple[str, str]: Food label, and prediction proba
    """
    test_image = get_test_data(image_path)
    model = get_model()
    pred = model.predict(test_image)
    pred_num = np.argmax(pred, axis=1)
    pred_percentage = str(round(pred[0][pred_num].item() * 100, 2))
    logging.info("Prediction completed...")
    return PRED_MAP[pred_num.item()], pred_percentage


def get_test_data(image_path: str) -> tf.keras.preprocessing.image.NumpyArrayIterator:
    """Accept an image via path and prepare a np array iterator
    This Iterator yielding data from a np array can then be passed into a TF model for prediction

    Args:
        image_path (str): Path to test image

    Returns:
        tf.keras.preprocessing.image.NumpyArrayIterator: To be passed into a TF model
    """
    logging.info("Uploading image...")
    size = (256, 256)
    test_image = Image.open(image_path)
    test_image = test_image.resize(size)
    test_image = np.expand_dims(test_image, axis=0)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    # return test_datagen.flow_from_directory(".", target_size=size, batch_size=32, class_mode='categorical', shuffle=False)
    return test_datagen.flow(test_image)


def main():
    """For local CLI testing - To run below command
    >> python -m src.inference your_test_image.png
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="input the image path.")
    args = parser.parse_args()
    pred = get_prediction(args.image_path)
    print("Test food prediction: ", pred)


if __name__ == "__main__":
    main()
