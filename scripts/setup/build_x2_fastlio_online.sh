#!/usr/bin/env bash
# Run in lxh only. Build a private ARM64 prefix; never install into the host SDK.
set -eo pipefail
run=${1:?private online localization directory}
[[ $(uname -m) == aarch64 ]] || { echo 'ARM64 build required' >&2; exit 1; }
[[ -f "$run/src/FAST_LIO_ROS2/src/laserMapping.cpp" ]]
[[ ! -e "$run/install/lib/fast_lio/fastlio_mapping" ]]
source /opt/ros/humble/setup.bash
python_library=$(/usr/bin/python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR")+"/"+sysconfig.get_config_var("LDLIBRARY"))')
python_include=$(/usr/bin/python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')
cmake -S "$run/src/livox_ros_driver2" -B "$run/build/interfaces" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$run/install" \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPython3_EXECUTABLE=/usr/bin/python3
cmake --build "$run/build/interfaces" -j2
cmake --install "$run/build/interfaces"
source "$run/install/share/livox_ros_driver2/local_setup.bash"
cmake -S "$run/src/FAST_LIO_ROS2" -B "$run/build/fast_lio" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$run/install" \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DPYTHON_INCLUDE_DIR="$python_include" -DPYTHON_LIBRARY="$python_library"
cmake --build "$run/build/fast_lio" -j2
cmake --install "$run/build/fast_lio"
file "$run/install/lib/fast_lio/fastlio_mapping"
sha256sum "$run/install/lib/fast_lio/fastlio_mapping"
