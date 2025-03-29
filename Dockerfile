# Use the official PyTorch container with CUDA support as the base image.
# Adjust the tag as needed; here we use PyTorch 2.0 with CUDA 11.7 and cuDNN 8.
# FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Build arguments for user and group IDs; defaults to 1000.
ARG USER_ID=1000
ARG GROUP_ID=1000

# Set environment variable to ensure non-interactive apt-get installs.
ENV DEBIAN_FRONTEND=noninteractive

# Install additional system dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    sudo \
    vim \
    tmux \
    htop \
    x11-apps \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version.
RUN pip install --upgrade pip

# Get some useful Python packages via pip
RUN pip install --no-cache-dir \
    matplotlib \
    tqdm \
    jupyterlab \
    normflows \
    gymnasium \
    stable-baselines3 \
    yapf \
    tensorboard \
    "gymnasium[mujoco]" \
    "gymnasium[classic-control]"

# Create a non-root user named "dev" with the provided UID/GID.
RUN groupadd -g ${GROUP_ID} dev && \
    useradd -m -u ${USER_ID} -g dev -s /bin/bash dev && \
    echo "dev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Create a workspace directory and change its ownership to the "dev" user.
RUN mkdir -p /home/dev/workspace && chown -R dev:dev /home/dev/workspace

# Set the working directory.
WORKDIR /home/dev/workspace

# Install this package in the container.
COPY . /home/dev/workspace/
RUN pip install -e .

# Declare a volume for the workspace to share code with the host.
VOLUME [ "/home/dev/workspace" ]

# Switch to the non-root user.
USER dev

# Set the default command to bash.
CMD ["bash"]
