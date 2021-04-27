# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_BUILD_TYPE Debug)

project(invisible_ai)

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(multiFeatureDetection src/multiFeatureDetection.cpp)
target_link_libraries (multiFeatureDetection ${OpenCV_LIBS})

add_executable(mfd-2 src/mfd-2.cpp)
target_link_libraries (mfd-2 ${OpenCV_LIBS})


