#!/bin/bash

conan install . --build=missing
cmake -B build -DCMAKE_TOOLCHAIN_FILE=build/Release/generators/conan_toolchain.cmake
cmake --build build

 ./build/main
