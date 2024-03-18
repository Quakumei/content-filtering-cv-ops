import os

import bentoml
import tensorflow_hub as hub
import tensorflow as tf

MODELS_DIR = "./models"
MODEL_FILES = {"299-inception-v3": "nsfw.299x299.h5"}
MODEL = "299-inception-v3"
MODEL_FILE = MODEL_FILES[MODEL]

TEST_DIR = "./data/test"
TEST_FILES: dict[str, str] = {
    "nsfw_dir": os.path.join(TEST_DIR, "nsfw"),
    "sfw_dir": os.path.join(TEST_DIR, "sfw"),
}

model_path = os.path.join(MODELS_DIR, MODEL_FILE)
model_tag = f"nsfw-{MODEL}"

def load_model(model_path):
    if model_path is None or not os.path.exists(model_path):
    	raise ValueError("saved_model_path must be the valid directory of a saved model to load.")
    # model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model(model_path)
bentoml.keras.save_model(model_tag, model)
print(f"{model_tag}")
