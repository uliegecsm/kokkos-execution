cmake_minimum_required(VERSION 4.0.0 FATAL_ERROR)

project(kokkos-execution VERSION 0.0.1 LANGUAGES CXX)

# Read the dependencies file.
file(READ dependencies.json DEPENDENCIES_JSON)

# Find Google Test.
string(JSON GTest_REQUIRED_VERSION GET "${DEPENDENCIES_JSON}" github googletest)

find_package(GTest ${GTest_REQUIRED_VERSION} CONFIG REQUIRED)

# Find Kokkos.
find_package(Kokkos REQUIRED COMPONENTS kokkoscore)

enable_testing()
add_subdirectory(tests)
