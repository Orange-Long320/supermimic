#!/usr/bin/env bash
# Build only in the supplied isolated server workspace. No system installation.
set -eo pipefail
workspace=${1:?Usage: build_x2_fastlio_offline.sh ISOLATED_WORKSPACE}
test "$(uname -s)" = Linux
test -f "$workspace/sources-build.tar.gz"
test ! -e "$workspace/src"
cd "$workspace"
printf '%s  %s\n' 2586007a7d816df288034f65dfa44c5d1734b605c75777809cb4ad98cf321cb0 sources-build.tar.gz | sha256sum -c -
tar -xzf sources-build.tar.gz
source /opt/ros/humble/setup.bash
cmake -S src/livox_ros_driver2 -B build/interfaces \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$workspace/install" \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPython3_EXECUTABLE=/usr/bin/python3
cmake --build build/interfaces -j2
cmake --install build/interfaces
source "$workspace/install/setup.bash"
cmake -S src/FAST_LIO_ROS2 -B build/fast_lio \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$workspace/install" \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DPYTHON_INCLUDE_DIR=/usr/include/python3.10 \
  -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.10.so
cmake --build build/fast_lio -j2
cmake --install build/fast_lio
date -Is
