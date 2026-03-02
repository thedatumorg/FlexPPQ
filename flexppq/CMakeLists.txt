add_library(
  PQ SHARED
  utils/IO.hpp
  utils/Heap.cpp
  PQ.cpp
  PartitionedPQ.cpp
)

target_include_directories(PQ PUBLIC
  ${PROJECT_SOURCE_DIR}/flexppq
  ${PROJECT_SOURCE_DIR}/external/eigen
)

FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

