import bentoml

import numpy as np
from bentoml.io import Image
from bentoml.io import JSON

MODEL_TAG = "nsfw-299-inception-v3"
IMG_DIM = 299
CATEGORIES = ["drawings", "hentai", "neutral", "porn", "sexy"]
NSFW_CATEGORIES = ["hentai", "porn", "sexy"]


def probs_to_verdict(
    probs, categories=CATEGORIES, nsfw_categories=NSFW_CATEGORIES
) -> bool:
    """Based on predicted probas, return the verdict, whether
    image is nsfw or not"""
    max_proba_idx = np.argmax(probs)
    cat = categories[max_proba_idx]
    return cat in nsfw_categories


runner = bentoml.keras.get(f"{MODEL_TAG}:latest").to_runner()
svc = bentoml.Service(f"{MODEL_TAG}", runners=[runner])


@svc.api(input=Image(), output=JSON())
async def predict(img):
    # Yes, imports *must* be there
    # Preprocessing
    img = img.resize((IMG_DIM, IMG_DIM))
    arr = np.array(img, dtype=np.float64)
    arr = arr[..., :3]
    arr = np.expand_dims(arr, axis=0)
    arr /= 255

    # Inference
    preds = await runner.async_run(arr)
    probs = preds[0]

    # NSFW-detector specific
    verdict = {"is_nsfw": probs_to_verdict(probs)}
    return verdict
