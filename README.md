# Active SLAM with Deep RL

## Setup

Working on the project requires Docker to be installed. You can install Docker on Linux with:

```bash
curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh
```

When working on an NVIDIA GPU, for Docker to be able to use your GPU, you must [install the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) as well.

Once Docker and the NVIDIA Container Toolkit are installed, build and run the Docker container with:
```bash
chmod +x run_container && ./run_container.sh
```