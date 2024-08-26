from nvcr.io/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

WORKDIR /root
USER root

ENV PATH /usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 서버 관련 유틸
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils && \
    apt-get install -y ffmpeg wget net-tools build-essential git curl vim nmon tmux lsof && \
    apt-get install -y python3.10 python3.10-dev python3.10-venv python3-pip

RUN ln -s /usr/bin/python3.10 /usr/bin/python

# 파이썬 관련 유틸
RUN pip install -U pip wheel setuptools && \
    pip install transformers accelerate datasets evaluate && \
    pip install trl peft bitsandbytes scipy sentencepiece deepspeed wandb pillow einops && \
    pip install ruff natsort setproctitle glances[gpu] && \
    pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 --index-url https://download.pytorch.org/whl/cu121
