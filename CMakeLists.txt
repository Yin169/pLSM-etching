cmake_minimum_required(VERSION 3.10)
project(DFISEParser)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    ${PROJECT_SOURCE_DIR}/src
)

# Homebrew package locations
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")
include_directories(/opt/homebrew/include)

# Find packages

set(TEST_DEPENDENCIES
)


add_executable(main main.cpp src/DFISEParser.cpp)
target_link_libraries(main PRIVATE ${TEST_DEPENDENCIES})
target_link_directories(main PRIVATE /opt/homebrew/lib)

