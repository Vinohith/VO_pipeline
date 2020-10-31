cmake_minimum_required(VERSION 3.1)
project(ba)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(G2O REQUIRED)

include_directories(
        ${G2O_INCLUDE_DIRS}
        "/usr/include/eigen3/"
)

add_executable(ba ba.cpp)
target_link_libraries(ba ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${G2O_TYPES_SBA} )