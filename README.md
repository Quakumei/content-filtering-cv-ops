# NSFW detector

BentoML packing for [nsfw-detector](https://pypi.org/project/nsfw-detector/) (a.k.a [nsfw_model](https://github.com/GantMan/nsfw_model/))

## Setup

```bash
# Fetch the model
./download.sh 

# Install the library (required only to load the keras model)
pip install nsfw-detector

```

## Test 

To check sanity and inference, first recreate
the following directory structure. 

```tree
data
└── test
    ├── nsfw
    └── sfw
```

Next, fill `nsfw` and `sfw` folders with test
images (around 20 for each is advised)
and then run `sanity_check.py`

```bash
python sanity_check.py
```

If you are willing to disable GPU inference, 
set environment variable `CUDA_VISIBLE_DEVICES=""`, e.g.: 

```bash
CUDA_VISIBLE_DEVICES="" python sanity_check.py
```

## Test (bentoml)

To check "readyiness" of model to be served via bentoml,
you can run `sanity_check_bentoml.py`. 

```bash
python sanity_check_bentoml.py
```

It completes a similar
task as in `sanity_check.py`, but fetches model from the 
bentoml model registry. Therefore, the model should be
uploaded to model registry beforehand by running 

```bash
python save_model.py
```

## Building a container

Before building a container you should [install bentoml](https://docs.bentoml.org/en/latest/quickstarts/install-bentoml.html#id1)
and fetch the model file. For the sake of convenience
you can download the model by running `download.sh` script.
It creates `models` folder and downloads the model 
for nsfw classification.

```bash
chmod +x download.sh
./download.sh
```

Also, you'll need `nsfw-detector` to load the model:

```bash
pip install nsfw-detector
```

Having done that, now you need to add the model to the bentoml registry:

```bash
python save_model.py
```

Lastly, to build the container, please run 

```bash
bentoml build --containerize
```

Voila - nsfw moderation is now ready to be shipped as any other docker container!


## I want to develop locally without containers

You can:

```bash
bentoml serve --reload --api-workers 1
```

## FAQ

- nsfw-detector / when loading model / failing sanity_check.py gives me an error about `tensorflow... something ...keras.tracking` or something in that manner.

The error is caused seemingly by some version mismatches? idk really, updating tensorflow-hub actually helped me.

```bash
# Despite causing dependency conflict, actually a fix
pip install tensorflow-hub==0.15.0
```
