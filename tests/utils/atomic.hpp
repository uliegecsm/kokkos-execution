#ifndef KOKKOS_EXECUTION_TESTS_UTILS_ATOMIC_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_ATOMIC_HPP

#include "Kokkos_Core.hpp"

#if defined(KOKKOS_ENABLE_CUDA)
#    include <cuda/atomic>
#elif defined(KOKKOS_ENABLE_HIP)
#    include "desul/atomics/Adapt_HIP.hpp"
#elif defined(KOKKOS_ENABLE_SYCL)
#    include "desul/atomics/Adapt_SYCL.hpp"
#endif

/**
 * @file
 *
 * User-chosen memory scope and order atomic operations
 * ----------------------------------------------------
 *
 * @c Kokkos uses @c desul for atomics and defaults to **device scope** on all backends.
 * That is sufficient when only GPU threads (of the same device) share a memory location, but yields
 * undefined behaviour when both CPU and GPU threads atomically access the same address
 * (e.g. a @c Kokkos::View in @c Kokkos::SharedSpace).
 *
 * This file provides a portable, atomic add with user-chosen memory scope and order.
 *
 * For more information:
 *  * https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#atomicity
 *  * https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives/atomic.html#concurrency-restrictions
 *  * https://llvm.org/docs/AMDGPUUsage.html#memory-scopes
 *  * https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html#atomic-functions
 */

#if defined(KOKKOS_ENABLE_CUDA)
namespace desul::Impl {
template <class MemoryScope>
struct CudaMemoryScope;

template <>
struct CudaMemoryScope<MemoryScopeDevice> {
    static constexpr auto value = cuda::thread_scope_device;
};

template <>
struct CudaMemoryScope<MemoryScopeSystem> {
    static constexpr auto value = cuda::thread_scope_system;
};

template <class MemoryOrder>
struct CudaMemoryOrder;

template <>
struct CudaMemoryOrder<MemoryOrderRelaxed> {
    static constexpr auto value = cuda::memory_order_relaxed;
};
} // namespace desul::Impl
#endif

namespace Tests::Utils {

//! Atomically add @p val to @c *ptr using given memory scope and order.
template <typename Scope, typename Order, typename T>
KOKKOS_FUNCTION void atomic_add(T* ptr, const T val) {
#if defined(KOKKOS_ENABLE_CUDA)
    cuda::atomic_ref<T, desul::Impl::CudaMemoryScope<Scope>::value>(*ptr)
        .fetch_add(val, desul::Impl::CudaMemoryOrder<Order>::value);
#elif defined(KOKKOS_ENABLE_HIP)
    __hip_atomic_fetch_add(
        ptr, val, desul::Impl::HIPMemoryOrder<Order>::value, desul::Impl::HIPMemoryScope<Scope>::value);
#elif defined(KOKKOS_ENABLE_SYCL)
    desul::Impl::sycl_atomic_ref<T, Order, Scope>{*ptr}.fetch_add(val);
#else
    static_assert(std::same_as<Order, desul::MemoryOrderRelaxed>);
    std::atomic_ref<T>(*ptr).fetch_add(val, std::memory_order_relaxed);
#endif
}

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_ATOMIC_HPP
