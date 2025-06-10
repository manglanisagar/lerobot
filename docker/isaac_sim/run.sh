#!/bin/bash

# Allow local X server access
xhost +

# Run the container with GUI and ROS 2 support
docker run --name isaac-sim \
  -it \
  --gpus all \
  --rm \
  --entrypoint='/bin/bash' \
  --network=host \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $HOME/.Xauthority:/root/.Xauthority \
  -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
  -v ~/docker/isaac-sim/documents:/root/Documents:rw \
  -v ~/projects/cs224r/project/code:/isaac-sim/code \
  -v ~/projects/cs224r/project/recordings:/isaac-sim/recordings \
  isaac_sim_ros2:4.5.0-Humble
