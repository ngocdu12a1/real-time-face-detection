cmake_minimum_required(VERSION 3.13)
project(helloDlib)

include_directories("/home/du/Documents/opensource/dlib-19.21/libs/include")
include_directories("/home/du/Documents/opensource/opencv-4.4.0/build/libs/include/opencv4")
include_directories("/usr/local/include/facedetection")

file(GLOB LIBSCV /home/du/Documents/opensource/opencv-4.4.0/build/libs/lib/*.so)

set(CMAKE_CXX_STANDARD 11)

add_executable(helloDlib main.cpp detector.h detector.cpp)


target_link_libraries(helloDlib
        /home/du/Documents/opensource/dlib-19.21/build/dlib/libdlib.so
        /home/du/Documents/opensource/OpenBLAS-0.3.10/install/linux/lib/libopenblas.so
        ${LIBSCV}
        /usr/local/lib/libfacedetection.so)