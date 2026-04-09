set -ex

COMPILER_FAMILY=$1
CMAKE_PRESET=$2

CPM_VERSION=$(python3 -c 'import json; print(json.load(open("dependencies.json"))["github"]["cpm"])')

PLOG_SOURCE_DIR=$PWD/external/plog

CMAKE_INSTALL_PREFIX=/opt/this-is-our-install-folder-for-test-purposes

echo "> Running install test for CMake preset ${CMAKE_PRESET}. Installation is expected to happen in ${CMAKE_INSTALL_PREFIX}."

cmake --install build-with-${CMAKE_PRESET} --prefix ${CMAKE_INSTALL_PREFIX}

export KokkosExecution_ROOT=${CMAKE_INSTALL_PREFIX}

echo "> Running find package test with installation expected in ${CMAKE_INSTALL_PREFIX}."

WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT

echo "> Working in ${WORK_DIR}."

cat << EOF > ${WORK_DIR}/CMakeLists.txt
cmake_minimum_required(VERSION 4.0.0)

project(test LANGUAGES CXX)

include(/opt/cpm-${CPM_VERSION}/CPM.cmake)

cpmaddpackage(
    NAME plog
    SOURCE_DIR "${PLOG_SOURCE_DIR}"
    OPTIONS "PLOG_BUILD_SAMPLES OFF" "PLOG_INSTALL OFF" "PLOG_BUILD_TESTS OFF"
)

find_package(KokkosExecution CONFIG REQUIRED)

add_executable(install_test)
target_sources(install_test PRIVATE install_test.cpp)
target_link_libraries(install_test PRIVATE Kokkos::Execution)
EOF

cat << EOF > ${WORK_DIR}/install_test.cpp
#include "Kokkos_Core.hpp"

#include "kokkos-execution/execution_space.hpp"
#include "kokkos-execution/graph.hpp"

int main() {
    const Kokkos::ScopeGuard guard{};
    {
        const Kokkos::DefaultExecutionSpace exec {};
        const Kokkos::Execution::ExecutionSpaceContext ctx{exec};
        stdexec::sync_wait(stdexec::schedule(ctx.get_scheduler()) | stdexec::then(KOKKOS_LAMBDA(){}));
    }
    {
        const Kokkos::DefaultExecutionSpace exec {};
        const Kokkos::Execution::GraphContext ctx{exec};
        stdexec::sync_wait(stdexec::schedule(ctx.get_scheduler()));
    }
    return EXIT_SUCCESS;
}
EOF

cd "$WORK_DIR"

if [ "${COMPILER_FAMILY}" = "clang" ];then
    export CXX=clang++
elif [ "${COMPILER_FAMILY}" = "rocm" ];then
    export CXX=hipcc
elif [ "${COMPILER_FAMILY}" = "intel" ];then
    export CXX=icpx
fi

cmake -S . -B build --warn-uninitialized

cmake --build build -j4 --verbose

./build/install_test

echo "> Install and test success."
