cmake_minimum_required(VERSION 3.13)

project(FlexPPQ)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Wno-psabi -Wno-sign-compare")

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
if(NOT COMPILER_SUPPORTS_AVX2)
    message(FATAL_ERROR "Error: Your compiler does not support AVX2 instructions (-mavx2).")
endif()


message("Default OPT LEVEL")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -ffast-math -mfma -funroll-loops")
add_definitions(-march=native)

include_directories(external/eigen)
include_directories(flexppq)

find_library(LAPACK_LIB NAMES lapack PATHS ${CMAKE_SOURCE_DIR}/external/lapack)

if (LAPACK_LIB)
    message("LAPACKE library found: ${LAPACK_LIB}")
else()
    message(FATAL_ERROR "LAPACK library not found!")
endif()

link_directories(external/lapack)

add_subdirectory(external/eigen)
add_subdirectory(flexppq)
add_subdirectory(examples)

add_definitions(-DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE)
add_definitions(-DEIGEN_DONT_PARALLELIZE)

FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()
