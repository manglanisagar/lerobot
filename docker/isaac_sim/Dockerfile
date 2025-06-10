# Isaac Sim base image
FROM nvcr.io/nvidia/isaac-sim:4.5.0

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    lsb-release \
    gnupg2 \
    locales \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Locale setup
RUN locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

# Add ROS 2 apt source securely (no apt-key)
RUN mkdir -p /usr/share/keyrings && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2.list

# Install ROS 2 core and robot simulation packages
RUN apt-get update && apt-get install -y \
    ros-humble-ros-base \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install ROS 2 core and robot simulation packages
RUN apt-get update && apt-get install -y \
    ros-humble-joint-state-publisher-gui \
    ros-humble-xacro \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros2-control \
    ros-humble-ros-ign-bridge \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    ros-humble-nav2-util \
    ros-humble-nav2-msgs \
    ros-humble-nav2-lifecycle-manager \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV ROS_DISTRO=humble
ENV ROS_VERSION=2
ENV ROS_PYTHON_VERSION=3
ENV OMNI_KIT_ALLOW_ROOT=1

COPY entrypoint.sh /isaac-sim/entrypoint.sh
RUN chmod +x /isaac-sim/entrypoint.sh

# Setup persistent volumes
VOLUME ["/isaac-sim/kit/cache", "/root/.cache/ov", "/root/.cache/pip", "/root/.cache/nvidia/GLCache", "/root/.nv/ComputeCache", "/root/.nvidia-omniverse/logs", "/root/.local/share/ov/data", "/root/Documents"]

# Set shell to bash and source ROS on login
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Default CMD (optional): you can uncomment this to auto-launch Isaac Sim
ENTRYPOINT ["/isaac-sim/entrypoint.sh"]
