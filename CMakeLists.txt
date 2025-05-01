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


# Add your executable
add_executable(levelset src/LevelSetMethod.cpp)
target_link_libraries(levelset ${EIGEN3_LIBRARIES} ${CGAL_LIBRARIES})

# add_executable(vdbLevelset src/OpenVDBLevelSet.cpp src/OpenVDBLevelSet.hpp)
# target_link_libraries(vdbLevelset ${CGAL_LIBRARIES} ${EIGEN3_LIBRARIES} ${OpenVDB_LIBRARIES})

# Add your executable
add_executable(alphawrap src/alphawrap.cpp src/output_helper.h)
target_link_libraries(alphawrap ${CGAL_LIBRARIES})
