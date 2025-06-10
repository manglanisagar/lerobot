#!/bin/bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=/isaac-sim/exts/isaacsim.ros2.bridge/humble/lib:$LD_LIBRARY_PATH
export AMENT_PREFIX_PATH=/isaac-sim/exts/isaacsim.ros2.bridge/humble

# Optional: print them for debug
echo "[INFO] Launching Isaac Sim with ROS 2 bridge env vars"
echo "  RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "  AMENT_PREFIX_PATH=$AMENT_PREFIX_PATH"

exec ./isaac-sim.sh "$@"
