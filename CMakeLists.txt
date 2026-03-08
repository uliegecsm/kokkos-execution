cmake_minimum_required(VERSION 4.0.0 FATAL_ERROR)

project(kokkos-execution VERSION 0.0.1 LANGUAGES CXX)

# Options.
option(KokkosExecution_ENABLE_CLANG_TIDY "Enable clang-tidy checks" OFF)

# Read the dependencies file.
file(READ dependencies.json DEPENDENCIES_JSON)

# Add Plog logging library.
include(cmake/modules/BuildPlog.cmake)

# Find Google Test.
string(JSON GTest_REQUIRED_VERSION GET "${DEPENDENCIES_JSON}" github googletest)

find_package(GTest ${GTest_REQUIRED_VERSION} CONFIG REQUIRED)

# Find Kokkos.
find_package(Kokkos REQUIRED COMPONENTS kokkoscore)

if(KokkosExecution_ENABLE_CLANG_TIDY)
  # Setup clang-tidy configuration.
  configure_file(${CMAKE_SOURCE_DIR}/.clang-tidy ${CMAKE_BINARY_DIR}/.clang-tidy @ONLY)
  set(CMAKE_CXX_CLANG_TIDY "clang-tidy" "--config-file=${CMAKE_BINARY_DIR}/.clang-tidy")
endif()

enable_testing()
add_subdirectory(tests)
