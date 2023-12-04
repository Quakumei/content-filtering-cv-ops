# NSFW detector

Container for [nsfw-detector](https://pypi.org/project/nsfw-detector/) (a.k.a [nsfw_model](https://github.com/GantMan/nsfw_model/))

## Setup

```bash
# Fetch the model
./download.sh 

# Install the library
pip install nsfw-detector

# Despite causing dependency conflict, actually a fix
pip install tensorflow-hub==0.15.0
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
