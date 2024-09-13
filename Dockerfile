FROM nvcr.io/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

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
    pip install --no-cache-dir git+https://github.com/huggingface/transformers.git@781bbc4d980abd2b21c332fd3122b733dba35d10 && \
    pip install accelerate==0.34.2 datasets==2.21.0 evaluate==0.4.2 trl==0.9.6 peft==0.12.0 deepspeed==0.15.0 && \
    pip install bitsandbytes==0.43.3 scipy==1.14.1 sentencepiece==0.2.0 pillow==10.4.0 liger-kernel==0.2.1 && \
    pip install ruff natsort setproctitle glances[gpu] wandb comet-ml

RUN pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install flash-attn==2.6.3
