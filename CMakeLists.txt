# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_BUILD_TYPE Debug)

project(invisible_ai)

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
#add_subdirectory(src)
#add_executable(playVideo src/playVideo.cpp)
#target_link_libraries (playVideo ${OpenCV_LIBS})
#add_executable(SIFT src/SIFT.cpp)
#target_link_libraries (SIFT ${OpenCV_LIBS})
#add_executable(cornerHarris src/cornerHarris.cpp)
#target_link_libraries (cornerHarris ${OpenCV_LIBS})
add_executable(multiFeatureDetection src/multiFeatureDetection.cpp)
target_link_libraries (multiFeatureDetection ${OpenCV_LIBS})
#add_executable(canny src/canny.cpp)
#target_link_libraries (canny ${OpenCV_LIBS})

#target_link_libraries(DisplayImage ${OpenCV_LIBS} )

