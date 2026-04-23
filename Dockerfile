# MapAttack Artifact Evaluation Environment
FROM nvidia/cuda:11.1-devel-ubuntu20.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    git \
    wget \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.8 as default python
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# Install PyTorch with CUDA support
RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Set working directory
WORKDIR /workspace/mapattack

# Copy requirements and install Python dependencies
COPY requirements_complete.txt .
RUN pip install -r requirements_complete.txt

# Copy source code
COPY . .

# Install mmdetection3d
RUN cd mmdetection3d && python setup.py develop

# Install geometric kernel attention
RUN cd projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn && \
    python setup.py build install

# Set environment variables
ENV PYTHONPATH=/workspace/mapattack:$PYTHONPATH

# Create directories for data and outputs
RUN mkdir -p /workspace/data /workspace/outputs

# Default command
CMD ["/bin/bash"]
