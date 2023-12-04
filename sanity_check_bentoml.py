import os

import pandas as pd

import bentoml
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

model_tag = f"nsfw-{MODEL}"


def test_model_lib_predict(model, test_files: dict):
    """Test model via nsfw_detector lib"""
    res_nsfw = nsfw_predict.classify(model, test_files["nsfw_dir"])
    res_sfw = nsfw_predict.classify(model, test_files["sfw_dir"])
    df_nsfw = (
        pd.DataFrame(res_nsfw)
        .T.reset_index()
        .sort_values("index")
        .reset_index()
        .round(2)
    )
    df_sfw = (
        pd.DataFrame(res_sfw)
        .T.reset_index()
        .sort_values("index")
        .reset_index()
        .round(2)
    )
    df_nsfw["index"] = df_nsfw["index"].apply(lambda x: x.split("/")[-1])
    df_sfw["index"] = df_sfw["index"].apply(lambda x: x.split("/")[-1])
    return df_nsfw, df_sfw


# Test saved model
model = bentoml.keras.load_model(f"{model_tag}:latest")
nsfw_df, sfw_df = test_model_lib_predict(model, TEST_FILES)
print(nsfw_df)
print(sfw_df)
