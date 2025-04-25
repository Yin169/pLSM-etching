cmake_minimum_required(VERSION 3.10)
project(EDA_competition)

# Find VTK package
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

# Find PCL package
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(CGAL REQUIRED)


# # Add your executable
# add_executable(main src/main.cpp src/DFISEParser.cpp)

# # Link against VTK and PCL libraries
# target_link_libraries(main ${VTK_LIBRARIES} ${PCL_LIBRARIES})

# Add your executable
add_executable(alphawrap src/alphawrap.cpp src/output_helper.h)

# Link against VTK and PCL libraries
target_link_libraries(alphawrap ${CGAL_LIBRARIES})