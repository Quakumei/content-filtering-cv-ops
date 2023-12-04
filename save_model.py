import os

import bentoml
import pandas as pd

from nsfw_detector import predict as nsfw_predict

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


def load_model(h5_path, image_dim: int = 299):
    """Achtung: changes nsfw_predict constant"""
    nsfw_predict.IMAGE_DIM = image_dim
    return nsfw_predict.load_model(h5_path)


model = load_model(model_path)
bentoml.keras.save_model(model_tag, model)
print(f"{model_tag}")
