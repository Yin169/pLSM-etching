cmake_minimum_required(VERSION 3.10)
project(EDA_competition)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find VTK package
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

# Find PCL package
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(CGAL REQUIRED)
find_package(Eigen3 REQUIRED)
set(CMAKE_MODULE_PATH "/opt/homebrew/lib/cmake/OpenVDB" ${CMAKE_MODULE_PATH})
find_package(OpenVDB REQUIRED)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Add your executable
add_executable(levelset src/LevelSetMethod.cpp)
target_link_libraries(levelset ${EIGEN3_LIBRARIES} ${CGAL_LIBRARIES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(levelset OpenMP::OpenMP_CXX)
endif()


# add_executable(vdbLevelset src/OpenVDBLevelSet.cpp src/OpenVDBLevelSet.hpp)
# target_link_libraries(vdbLevelset ${CGAL_LIBRARIES} ${EIGEN3_LIBRARIES} ${OpenVDB_LIBRARIES})

