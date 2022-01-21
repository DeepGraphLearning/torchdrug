FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

RUN apt-get update && \
    apt-get install -y libxrender1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html  && \
    pip install torchdrug