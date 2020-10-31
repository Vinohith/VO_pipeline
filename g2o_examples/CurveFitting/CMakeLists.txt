cmake_minimum_required(VERSION 3.1)
project(curvefitting)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(G2O REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
		${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        "/usr/include/eigen3/"
)

add_executable(curvefitting curvefitting.cpp)
target_link_libraries(curvefitting ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${OpenCV_LIBS})