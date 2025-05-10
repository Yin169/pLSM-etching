cmake_minimum_required(VERSION 3.10)
project(EDA_competition)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(CGAL REQUIRED)
find_package(Eigen3 REQUIRED)

# Find and configure OpenMP for multi-threading support
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(levelset src/levelset_main.cpp)
target_link_libraries(levelset ${EIGEN3_LIBRARIES} ${CGAL_LIBRARIES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(levelset OpenMP::OpenMP_CXX)
endif()


