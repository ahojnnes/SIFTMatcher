cmake_minimum_required(VERSION 2.8)


project(SIFT_MATCHER)


set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(Eigen3 REQUIRED)
find_package(OpenCL REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

include_directories(
    .
    ${EIGEN3_INCLUDE_DIR}
    ${OPENCL_INCLUDE_DIR}
)

add_library(sift_matcher sift_matcher.cc)

add_executable(usage usage.cc)
target_link_libraries(usage
    ${OPENCL_LIBRARIES}
    sift_matcher
)
