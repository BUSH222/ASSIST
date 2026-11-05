cmake_minimum_required(VERSION 3.20...4.4)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ASSIST_NATIVE "Add -march=native (local builds only, never for CI or wheels)" OFF)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_dsp cpp/bindings.cpp cpp/synth.cpp cpp/frontend.cpp cpp/modem.cpp cpp/conv.cpp)
target_include_directories(_dsp PRIVATE cpp)
# pocketfft (header only, BSD-3): SYSTEM keeps its warnings out of ours. The kernels are single-threaded (parallel
# rendering uses worker processes, plan §4.5), so pocketfft's own thread pool is compiled out; its plan cache
# (64 lengths) is kept.
target_include_directories(_dsp SYSTEM PRIVATE cpp/third_party)
target_compile_definitions(_dsp PRIVATE POCKETFFT_NO_MULTITHREADING POCKETFFT_CACHE_SIZE=64)

if(NOT MSVC)
  # No -ffast-math and no FMA contraction: float results stay predictable (plan §2).
  target_compile_options(_dsp PRIVATE -O3 -ffp-contract=off -Wall -Wextra)
  if(ASSIST_NATIVE)
    target_compile_options(_dsp PRIVATE -march=native)
    target_compile_definitions(_dsp PRIVATE ASSIST_NATIVE_BUILD)
  endif()
endif()

install(TARGETS _dsp LIBRARY DESTINATION assist)
