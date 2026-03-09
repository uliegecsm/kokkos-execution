cmake_minimum_required(VERSION 4.0.0 FATAL_ERROR)

project(kokkos-execution VERSION 0.0.1 LANGUAGES CXX)

# Options.
option(KokkosExecution_ENABLE_CLANG_TIDY "Enable clang-tidy checks" OFF)
option(KokkosExecution_ENABLE_DEBUG_LOGGING "Enable debug logging" OFF)
option(KokkosExecution_ENABLE_TESTS "Enable testing" ON)

# Read the dependencies file.
file(READ dependencies.json DEPENDENCIES_JSON)

# Enable further debugging by exporting compile commands.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find Kokkos.
find_package(Kokkos REQUIRED COMPONENTS kokkoscore)

if(KokkosExecution_ENABLE_CLANG_TIDY)
  # Setup clang-tidy configuration.
  configure_file(${CMAKE_SOURCE_DIR}/.clang-tidy ${CMAKE_BINARY_DIR}/.clang-tidy @ONLY)
  set(CMAKE_CXX_CLANG_TIDY "clang-tidy" "--config-file=${CMAKE_BINARY_DIR}/.clang-tidy")
endif()

# Find stdexec. Look for the copy of stdexec bundled as a submodule in this repository. The stdexec repository is the
# reference implementation for the C++26 additions to std::execution. It changes frequently. The submodule SHA thus
# records the exact version that this repository currently works with.
string(JSON CPM_VERSION GET "${DEPENDENCIES_JSON}" "github" "cpm")
include(/opt/cpm-${CPM_VERSION}/CPM.cmake)

cpmaddpackage(
  NAME
  stdexec
  SOURCE_DIR
  "${CMAKE_SOURCE_DIR}/external/stdexec"
  OPTIONS
  "STDEXEC_BUILD_TESTS OFF"
  "STDEXEC_BUILD_EXAMPLES OFF"
)

if(KokkosExecution_ENABLE_DEBUG_LOGGING OR KokkosExecution_ENABLE_TESTS)
  # Add Plog logging library.
  include(cmake/modules/BuildPlog.cmake)
endif()

# Build the Kokkos::Execution library.
add_library(kokkosexecution INTERFACE)
add_library(Kokkos::Execution ALIAS kokkosexecution)

target_sources(
  kokkosexecution
  INTERFACE FILE_SET
            headers
            TYPE
            HEADERS
            BASE_DIRS
            include/kokkos-execution
            FILES
            include/kokkos-execution/execution_space.hpp
            include/kokkos-execution/parallel_for.hpp
            include/kokkos-execution/execution_space/bulk.hpp
            include/kokkos-execution/execution_space/continues_on.hpp
            include/kokkos-execution/execution_space/context_fwd.hpp
            include/kokkos-execution/execution_space/domain.hpp
            include/kokkos-execution/execution_space/get_exec.hpp
            include/kokkos-execution/execution_space/operation_state.hpp
            include/kokkos-execution/execution_space/parallel_for.hpp
            include/kokkos-execution/execution_space/schedule_from.hpp
            include/kokkos-execution/execution_space/scoped_region.hpp
            include/kokkos-execution/execution_space/sync_wait.hpp
            include/kokkos-execution/execution_space/then.hpp
            include/kokkos-execution/impl/bulk.hpp
            include/kokkos-execution/impl/completion_signatures.hpp
            include/kokkos-execution/impl/env.hpp
            include/kokkos-execution/impl/sync_wait.hpp
            include/kokkos-execution/utils/ignore_warnings.hpp
)

target_include_directories(
  kokkosexecution INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/> $<INSTALL_INTERFACE:include/>
)

target_compile_features(kokkosexecution INTERFACE cxx_std_20)

target_compile_definitions(kokkosexecution INTERFACE KOKKOS_IMPL_PUBLIC_INCLUDE)

target_link_libraries(kokkosexecution INTERFACE Kokkos::kokkoscore STDEXEC::stdexec)

if(KokkosExecution_ENABLE_DEBUG_LOGGING)
  target_compile_definitions(kokkosexecution INTERFACE KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
  target_link_libraries(kokkosexecution INTERFACE plog::plog)
endif()

# Testing.
if(KokkosExecution_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
