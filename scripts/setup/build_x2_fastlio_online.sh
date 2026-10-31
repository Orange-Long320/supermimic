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
# Mounted ROS exports contain the host's absolute NumPy include path. Correct
# only those imported target properties in this private CMake invocation.
numpy_include=$(/usr/bin/python3 -c 'import numpy; print(numpy.get_include())')
cat > "$run/sdk_numpy_headers.cmake" <<'CMAKE'
function(sm_fix_sdk_numpy_headers)
  get_property(imported DIRECTORY PROPERTY IMPORTED_TARGETS)
  foreach(target IN LISTS imported)
    get_target_property(paths "${target}" INTERFACE_INCLUDE_DIRECTORIES)
    if(paths)
      list(TRANSFORM paths REPLACE "/usr/local/lib/python3.10/dist-packages/numpy/core/include" "${SM_NUMPY_INCLUDE}")
      set_property(TARGET "${target}" PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${paths}")
    endif()
  endforeach()
endfunction()
cmake_language(DEFER CALL sm_fix_sdk_numpy_headers)
CMAKE
cmake -S "$run/src/livox_ros_driver2" -B "$run/build/interfaces" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$run/install" \
  -DCMAKE_PROJECT_INCLUDE="$run/sdk_numpy_headers.cmake" -DSM_NUMPY_INCLUDE="$numpy_include" \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPython3_EXECUTABLE=/usr/bin/python3
cmake --build "$run/build/interfaces" -j2
cmake --install "$run/build/interfaces"
source "$run/install/share/livox_ros_driver2/local_setup.bash"
cmake -S "$run/src/FAST_LIO_ROS2" -B "$run/build/fast_lio" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$run/install" \
  -DCMAKE_PROJECT_INCLUDE="$run/sdk_numpy_headers.cmake" -DSM_NUMPY_INCLUDE="$numpy_include" \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DPYTHON_INCLUDE_DIR="$python_include" -DPYTHON_LIBRARY="$python_library"
cmake --build "$run/build/fast_lio" -j2
cmake --install "$run/build/fast_lio"
file "$run/install/lib/fast_lio/fastlio_mapping"
sha256sum "$run/install/lib/fast_lio/fastlio_mapping"
