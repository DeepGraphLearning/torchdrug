Docker
======

Run a Container
---------------

We prepare a pre-built docker image on Docker Hub, based on PyTorch 1.8.1 and CUDA 11.1.
To start a container with our docker image, use the following line.

```bash
docker run --gpus all -it milagraph/torchdrug 
```

Build Docker Image
------------------

You may also use the Dockerfile provided in this folder to build an image from scratch.
Run the following command in this directory.

```bash
docker build -t "your-image-name" .
```