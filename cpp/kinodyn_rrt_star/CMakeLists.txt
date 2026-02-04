cmake_minimum_required(VERSION 3.12)
project(kinodyn_rrt_star LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Eigen
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)

# Include dirs
include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Library sources
set(KINODYN_SOURCES
    src/kinodyn_rrt_star.cpp
    src/kdtree.cpp
)

add_library(kinodyn_rrt_star STATIC ${KINODYN_SOURCES})

target_include_directories(kinodyn_rrt_star PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(kinodyn_rrt_star PUBLIC Eigen3::Eigen Threads::Threads)

# Executable
add_executable(kino_rrt_planner app/main.cpp)

target_link_libraries(kino_rrt_planner PRIVATE kinodyn_rrt_star)

# Output dirs
set_target_properties(kino_rrt_planner PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
