add_definitions(-DEIGEN_DONT_PARALLELIZE)

include_directories(${PROJECT_SOURCE_DIR}/flexppq)
include_directories(${PROJECT_SOURCE_DIR}/external/eigen)

FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()


add_executable(demo "demo.cpp")
target_link_libraries(demo PRIVATE PQ)
target_link_libraries(demo PRIVATE -lglpk)
target_link_libraries(demo PRIVATE -lblas -llapacke)
target_link_libraries(demo PRIVATE -larmadillo)
