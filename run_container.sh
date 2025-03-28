# Set the image name
IMAGE_NAME="active-slam-rl"

# Build the Docker image with the current user's UID and GID
echo "Building Docker image '${IMAGE_NAME}'..."
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --network=host -t ${IMAGE_NAME} .
if [ $? -ne 0 ]; then
    echo "Docker build failed!"
    exit 1
fi

# Get the full path of the current directory to mount as a volume
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_DIR="$(realpath "$SCRIPT_DIR")"

echo "Using host directory ${HOST_DIR} as the shared workspace."

# Run the container with GPU support enabled (if available)
echo "Running container..."

xhost +local:

docker run --gpus all -it --rm \
    -v ${HOST_DIR}:/home/dev/workspace \
    --name ${IMAGE_NAME} \
    --network host \
    --ipc=host \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/.vscode-server:/root/.vscode-server \
    -p 8888:8888 \
    ${IMAGE_NAME}

xhost -local:
