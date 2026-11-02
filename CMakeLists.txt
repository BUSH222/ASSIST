cmake_minimum_required(VERSION 3.20...4.4)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ASSIST_NATIVE "Add -march=native (local builds only, never for CI or wheels)" OFF)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_dsp cpp/bindings.cpp)
target_include_directories(_dsp PRIVATE cpp)

if(NOT MSVC)
  # No -ffast-math and no FMA contraction: float results stay predictable (plan §2).
  target_compile_options(_dsp PRIVATE -O3 -ffp-contract=off -Wall -Wextra)
  if(ASSIST_NATIVE)
    target_compile_options(_dsp PRIVATE -march=native)
  endif()
endif()

install(TARGETS _dsp LIBRARY DESTINATION assist)
