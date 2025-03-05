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
    pip install git+https://github.com/huggingface/transformers.git@66f29aaaf55c8fe0c3dbcd24beede2ca4effac56 && \
    pip install accelerate datasets evaluate trl peft deepspeed liger-kernel && \
    pip install bitsandbytes scipy sentencepiece pillow && \
    pip install ruff natsort setproctitle glances[gpu] wandb cmake

RUN pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install flash-attn==2.7.2.post1
