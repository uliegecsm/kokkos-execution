cmake_minimum_required(VERSION 4.0.0 FATAL_ERROR)

set(KokkosExecution_VERSION_MAJOR 0)
set(KokkosExecution_VERSION_MINOR 0)
set(KokkosExecution_VERSION_PATCH 1)
set(KokkosExecution_VERSION
    "${KokkosExecution_VERSION_MAJOR}.${KokkosExecution_VERSION_MINOR}.${KokkosExecution_VERSION_PATCH}"
)

project(kokkos-execution VERSION ${KokkosExecution_VERSION} LANGUAGES CXX)

# Options.
option(KokkosExecution_ENABLE_CLANG_TIDY "Enable clang-tidy checks" OFF)
option(KokkosExecution_ENABLE_DEBUG_LOGGING "Enable debug logging" OFF)
option(KokkosExecution_ENABLE_DOCS "Enable documentation" OFF)
option(KokkosExecution_ENABLE_EXTRA_TYPE_CHECKING "Enable extra type checking, but increases compile time" OFF)
option(KokkosExecution_ENABLE_TESTS "Enable testing" ON)

include(CMakeDependentOption)
cmake_dependent_option(
  KokkosExecution_ENABLE_EVENT_DISPATCH "Enable event dispatch à la Kokkos Tools" ON KokkosExecution_ENABLE_TESTS OFF
)

foreach(KE_OPTION IN ITEMS CLANG_TIDY DEBUG_LOGGING DOCS TESTS EVENT_DISPATCH)
  message(STATUS "KokkosExecution_ENABLE_${KE_OPTION} is ${KokkosExecution_ENABLE_${KE_OPTION}}.")
endforeach()

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

if(SYCL IN_LIST Kokkos_DEVICES AND CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  include(cmake/functions/patch.cmake)
  apply_patch(${CMAKE_SOURCE_DIR}/external/stdexec-icpx-ice-when-all.patch ${CMAKE_SOURCE_DIR}/external/stdexec)
endif()

cpmaddpackage(
  NAME
  stdexec
  SOURCE_DIR
  "${CMAKE_SOURCE_DIR}/external/stdexec"
  OPTIONS
  "STDEXEC_BUILD_TESTS OFF"
  "STDEXEC_BUILD_EXAMPLES OFF"
  "STDEXEC_INSTALL ON"
  "STDEXEC_ENABLE_EXTRA_TYPE_CHECKING ${KokkosExecution_ENABLE_EXTRA_TYPE_CHECKING}"
)

# Compiler warnings.
include(cmake/modules/warnings.cmake)

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
            ${CMAKE_CURRENT_SOURCE_DIR}/
            FILES
            kokkos-execution/execution_space.hpp
            kokkos-execution/graph.hpp
            kokkos-execution/parallel_for.hpp
            kokkos-execution/stdexec.hpp
            kokkos-execution/execution_space/bulk.hpp
            kokkos-execution/execution_space/continues_on.hpp
            kokkos-execution/execution_space/domain.hpp
            kokkos-execution/execution_space/env.hpp
            kokkos-execution/execution_space/execution_space_fwd.hpp
            kokkos-execution/execution_space/get_exec.hpp
            kokkos-execution/execution_space/operation_state.hpp
            kokkos-execution/execution_space/parallel_for.hpp
            kokkos-execution/execution_space/schedule_from.hpp
            kokkos-execution/execution_space/scoped_region.hpp
            kokkos-execution/execution_space/sender_concepts.hpp
            kokkos-execution/execution_space/sync_wait.hpp
            kokkos-execution/execution_space/then.hpp
            kokkos-execution/execution_space/when_all.hpp
            kokkos-execution/graph/domain.hpp
            kokkos-execution/graph/events.hpp
            kokkos-execution/graph/graph_fwd.hpp
            kokkos-execution/graph/sender_concepts.hpp
            kokkos-execution/graph/sync_wait.hpp
            kokkos-execution/impl/attributes.hpp
            kokkos-execution/impl/bulk.hpp
            kokkos-execution/impl/completion_signatures.hpp
            kokkos-execution/impl/Cuda/event.hpp
            kokkos-execution/impl/dispatch_label.hpp
            kokkos-execution/impl/domain.hpp
            kokkos-execution/impl/env.hpp
            kokkos-execution/impl/event.hpp
            kokkos-execution/impl/HIP/event.hpp
            kokkos-execution/impl/HPX/event.hpp
            kokkos-execution/impl/immovable.hpp
            kokkos-execution/impl/receiver.hpp
            kokkos-execution/impl/sender_concepts.hpp
            kokkos-execution/impl/sender_introspection.hpp
            kokkos-execution/impl/SYCL/event.hpp
            kokkos-execution/impl/state.hpp
            kokkos-execution/impl/sync_wait.hpp
            kokkos-execution/utils/ignore_warnings.hpp
)

target_compile_features(kokkosexecution INTERFACE cxx_std_20)

target_compile_definitions(kokkosexecution INTERFACE KOKKOS_IMPL_PUBLIC_INCLUDE)

target_link_libraries(kokkosexecution INTERFACE Kokkos::kokkoscore STDEXEC::stdexec)

if(KokkosExecution_ENABLE_DEBUG_LOGGING)
  target_compile_definitions(kokkosexecution INTERFACE KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
  target_link_libraries(kokkosexecution INTERFACE $<BUILD_INTERFACE:plog::plog>)
endif()

if(KokkosExecution_ENABLE_EVENT_DISPATCH)
  find_package(KokkosUtils CONFIG REQUIRED)
  target_compile_definitions(kokkosexecution INTERFACE KOKKOS_EXECUTION_ENABLE_EVENT_DISPATCH)
  target_link_libraries(kokkosexecution INTERFACE $<BUILD_INTERFACE:Kokkos::kokkosutils>)
endif()

# Install.
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

configure_package_config_file(
  cmake/KokkosExecutionConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/KokkosExecutionConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KokkosExecution
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/KokkosExecutionConfigVersion.cmake" VERSION "${KokkosExecution_VERSION}"
  COMPATIBILITY ExactVersion
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/KokkosExecutionConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/KokkosExecutionConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KokkosExecution
)

install(
  TARGETS kokkosexecution
  EXPORT KokkosExecutionTargets
  FILE_SET headers
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
  EXPORT KokkosExecutionTargets
  FILE KokkosExecutionTargets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KokkosExecution
  NAMESPACE Kokkos::
)

if(KokkosExecution_ENABLE_TESTS OR KokkosExecution_ENABLE_DOCS)
  # Find Google Test.
  string(JSON GTest_REQUIRED_VERSION GET "${DEPENDENCIES_JSON}" github googletest)

  find_package(GTest ${GTest_REQUIRED_VERSION} CONFIG REQUIRED)
endif()

# Testing.
if(KokkosExecution_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# Documentation.
if(KokkosExecution_ENABLE_DOCS)
  include(cmake/modules/doxygen.cmake)
  add_subdirectory(docs)
endif()
