cmake_minimum_required(VERSION 3.15...3.26)
project(gauss LANGUAGES CXX)

find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

execute_process(COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())" OUTPUT_VARIABLE NUMPY_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT EXISTS "${NUMPY_INCLUDE_DIR}")
    message(FATAL_ERROR "NumPy include directory not found!")
endif()

pybind11_add_module(gauss gauss.cpp)

target_include_directories(gauss PRIVATE "${NUMPY_INCLUDE_DIR}")

target_compile_features(gauss PRIVATE cxx_std_17)

install(TARGETS gauss DESTINATION .)
