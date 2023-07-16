FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

# metainformation
LABEL org.opencontainers.image.version = "0.2.1"
LABEL org.opencontainers.image.authors = "TorchDrug Team"
LABEL org.opencontainers.image.source = "https://github.com/DeepGraphLearning/torchdrug"
LABEL org.opencontainers.image.licenses = "Apache License 2.0"
LABEL org.opencontainers.image.base.name="docker.io/pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel"

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && \
    apt-get install -y libxrender1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install torch-scatter torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html && \
    pip install torchdrug
