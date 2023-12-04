import os
import time

import pandas as pd

from nsfw_detector import predict as nsfw_predict


class catchtime:
    # https://stackoverflow.com/questions/33987060/python-context-manager-that-measures-time
    def __init__(self, title: str = None):
        self.title = title if title else "Time"

    def __enter__(self):
        print(f"{self.title}: start...")
        self.start = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = time.perf_counter() - self.start
        self.readout = f"{self.title}: {self.time:.3f} seconds"
        print(self.readout)


def count_files(dir: str) -> int:
    """Return number of files in folder (nonrecursive)"""
    num_files = len(
        [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    )
    return num_files


if __name__ == "__main__":
    MODELS_DIR = "./models"
    TEST_DIR = "./data/test"
    MODEL_FILES = {"299InceptionV3": "nsfw.299x299.h5"}
    MODEL = "299InceptionV3"
    MODEL_FILE = MODEL_FILES[MODEL]

    TEST_FILES: dict[str, str] = {
        "nsfw_dir": os.path.join(TEST_DIR, "nsfw"),
        "sfw_dir": os.path.join(TEST_DIR, "sfw"),
    }

    with catchtime(f"Model load - {MODEL}") as timer:
        nsfw_predict.IMAGE_DIM = 299
        model = nsfw_predict.load_model(os.path.join(MODELS_DIR, MODEL_FILE))

    total_file_count = sum([count_files(dir) for _, dir in TEST_FILES.items()])

    with catchtime(f"Test predictions - {total_file_count} files"):
        res_nsfw = nsfw_predict.classify(model, TEST_FILES["nsfw_dir"])
        res_sfw = nsfw_predict.classify(model, TEST_FILES["sfw_dir"])

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

    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)

    print("=" * 80)
    print("NSFW")
    print(df_nsfw)

    print("SFW")
    print(df_sfw)
