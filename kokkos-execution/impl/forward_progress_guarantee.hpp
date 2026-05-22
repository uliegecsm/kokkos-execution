#ifndef KOKKOS_EXECUTION_IMPL_FORWARD_PROGRESS_GUARANTEE_HPP
#define KOKKOS_EXECUTION_IMPL_FORWARD_PROGRESS_GUARANTEE_HPP

#include "kokkos-execution/stdexec.hpp"

#include "Kokkos_Concepts.hpp"

namespace Kokkos::Execution::Impl {
/**
 * @name Forward progress guarantee that a @c Kokkos backend provides for threads of execution
 *       generated for a given operation.
 *
 * By default, the weakest guarantee is given, *i.e.* @c stdexec::forward_progress_guarantee::weakly_parallel.
 * See also
 * https://github.com/NVIDIA/stdexec/blob/feff2aacf1d1f8470bce31442487fa557d7fdd76/include/stdexec/__detail/__schedulers.hpp#L493.
 *
 * References:
 *  - https://eel.is/c++draft/exec#get.fwd.progress
 *  - https://en.cppreference.com/cpp/language/multithread
 *  - https://eel.is/c++draft/intro.progress
 *  - https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/execution_model.html#device-threads
 *  - https://www.youtube.com/watch?v=g9Rgu6YEuqY&t=3600s
 *  - @cite pennycook-alignment-sycl-parallelism-with-cpp
 *  - @cite P3564R0
 *  - @cite lelbach-accelerated-cpp
 */
///@{
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KOKKOS_EXECUTION_IMPL_FORWARD_PROGRESS_GUARANTEE_OF(_exec_, _fpg_)                                             \
    template <>                                                                                                        \
    struct ForwardProgressGuaranteeOf<_exec_> {                                                                        \
        static constexpr auto value = stdexec::forward_progress_guarantee::_fpg_;                                      \
    };

template <Kokkos::ExecutionSpace Exec>
struct ForwardProgressGuaranteeOf {
    static constexpr auto value = stdexec::forward_progress_guarantee::weakly_parallel;
};

#if defined(KOKKOS_ENABLE_CUDA) && KOKKOS_IMPL_ARCH_NVIDIA_GPU >= 70
KOKKOS_EXECUTION_IMPL_FORWARD_PROGRESS_GUARANTEE_OF(Kokkos::Cuda, parallel)
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
KOKKOS_EXECUTION_IMPL_FORWARD_PROGRESS_GUARANTEE_OF(Kokkos::OpenMP, parallel)
#endif
#if defined(KOKKOS_ENABLE_SERIAL)
KOKKOS_EXECUTION_IMPL_FORWARD_PROGRESS_GUARANTEE_OF(Kokkos::Serial, parallel)
#endif
#if defined(KOKKOS_ENABLE_THREADS)
KOKKOS_EXECUTION_IMPL_FORWARD_PROGRESS_GUARANTEE_OF(Kokkos::Threads, parallel)
#endif

template <Kokkos::ExecutionSpace Exec>
constexpr stdexec::forward_progress_guarantee forward_progress_guarantee_of_v = ForwardProgressGuaranteeOf<Exec>::value;
///@}

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_FORWARD_PROGRESS_GUARANTEE_HPP
