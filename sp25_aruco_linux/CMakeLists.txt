
cmake_minimum_required(VERSION 3.16)
project(sp25_aruco_linux)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
find_package(OpenCV REQUIRED)
include_directories(include ${OpenCV_INCLUDE_DIRS} /opt/MVS/include)

add_executable(sp25_aruco
    src/main.cpp
    src/hik_camera.cpp
    src/aruco_detector.cpp
    src/robust_aruco_detector.cpp
    src/illumination_normalizer.cpp
    src/marker_quality.cpp
    src/marker_solver.cpp
)
target_link_libraries(sp25_aruco ${OpenCV_LIBS} /opt/MVS/lib/64/libMvCameraControl.so)

add_executable(tune_exposure
    src/tune_exposure.cpp
    src/hik_camera.cpp
    src/aruco_detector.cpp
    src/robust_aruco_detector.cpp
    src/illumination_normalizer.cpp
    src/marker_quality.cpp
)
target_link_libraries(tune_exposure ${OpenCV_LIBS} /opt/MVS/lib/64/libMvCameraControl.so)