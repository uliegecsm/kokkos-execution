#ifndef KOKKOS_EXECUTION_TESTS_UTILS_ATOMIC_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_ATOMIC_HPP

#include "Kokkos_Core.hpp"

#if defined(KOKKOS_ENABLE_CUDA)
#    include <cuda/atomic>
#endif

/**
 * @file
 *
 * System-scope atomic operations
 * ------------------------------
 *
 * @c Kokkos uses @c desul for atomics and defaults to **device scope** on all backends.
 * That is sufficient when only GPU threads (of the same device) share a memory location, but yields
 * undefined behaviour when both CPU and GPU threads atomically access the same address
 * (e.g. a @c Kokkos::View in @c Kokkos::SharedSpace).
 *
 * This file provides a portable, system-scope atomic add with relaxed memory order.
 *
 * For more information:
 *  * https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#atomicity
 *  * https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives/atomic.html#concurrency-restrictions
 *  * https://llvm.org/docs/AMDGPUUsage.html#memory-scopes
 *  * https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html#atomic-functions
 */

namespace Tests::Utils {

//! Atomically add @p val to @c *ptr using system scope and relaxed memory order.
template <typename T>
KOKKOS_FUNCTION void atomic_add(T* ptr, const T val) {
#if defined(KOKKOS_ENABLE_CUDA)
    cuda::atomic_ref<T, cuda::thread_scope_system>(*ptr).fetch_add(val, cuda::memory_order_relaxed);
#elif defined(KOKKOS_ENABLE_HIP)
    __hip_atomic_fetch_add(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    std::atomic_ref<T>(*ptr).fetch_add(val, std::memory_order_relaxed);
#endif
}

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_ATOMIC_HPP
