cmake_minimum_required(VERSION 3.5.0)
project(calibCamIntrinsic VERSION 0.1.0 LANGUAGES C CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
add_compile_options(-w)

# 
find_package(yaml-cpp REQUIRED)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(spdlog REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)

#
include_directories(
  include
  ${YAML_CPP_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

add_executable(calibCamIntrinsic 
  main.cpp
  src/chessboard/Chessboard.cc
  src/calib/CameraCalibration.cc
  src/camera_models/Camera.cc
  src/camera_models/CameraFactory.cc
  src/camera_models/CostFunctionFactory.cc
  src/camera_models/PinholeCamera.cc
  src/camera_models/PinholeFullCamera.cc
  src/camera_models/CataCamera.cc
  src/camera_models/EquidistantCamera.cc
  src/camera_models/ScaramuzzaCamera.cc
  src/sparse_graph/Transform.cc
  src/gpl/gpl.cc
  src/gpl/EigenQuaternionParameterization.cc
)

target_link_libraries(calibCamIntrinsic
  ${YAML_CPP_LIBRARIES}
  ${PCL_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CERES_LIBRARIES}
  spdlog::spdlog_header_only
  ${Boost_LIBRARIES}
)
