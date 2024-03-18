# NSFW detector

BentoML packing for [nsfw-detector](https://pypi.org/project/nsfw-detector/) (a.k.a [nsfw_model](https://github.com/GantMan/nsfw_model/))

## Building a container

I tested building a container on Python 3.11.2.

To build a container follow the steps below. 

Download the model in `models` dir by running:

```bash
chmod +x download.sh
./download.sh
```

Create virtual environment OUTSIDE of repo and install build dependencies required to create a container:

```bash
pip install -r requirements_build.txt
```

Having done that, now you need to add the model to the bentoml registry:

```bash
python save_model.py
```

Lastly, to build the container, please run 

```bash
bentoml build --containerize --version 0.0.1
```

The end of output should be like this:

```bash
 => exporting to image                                                                                                                                                                0.2s
 => => exporting layers                                                                                                                                                               0.2s
 => => writing image sha256:ab4b4ab06bd34305624b83054f050d4cd6be6ab204cf7506a9fd9393cadf2cb9                                                                                          0.0s
 => => naming to docker.io/library/nsfw-299-inception-v3:0.0.1
```

Note the tag in the end, check that it is the same in docker-compose.yml file. Then you can start the service with

```bash
docker-compose up
```
and access its OpenAPI docs via `http://localhost:3000`.


## FAQ

- nsfw-detector / when loading model / failing sanity_check.py gives me an error about `tensorflow... something ...keras.tracking` or something in that manner.

The error is caused seemingly by some version mismatches? idk really, updating tensorflow-hub actually helped me.

```bash
# Despite causing dependency conflict, actually a fix
pip install tensorflow-hub==0.15.0
```

- I want to develop locally without containers

You can:

```bash
bentoml serve --reload --api-workers 1
```
