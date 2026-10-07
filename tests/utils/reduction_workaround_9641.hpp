#ifndef KOKKOS_EXECUTION_TESTS_UTILS_REDUCTION_WORKAROUND_9641_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_REDUCTION_WORKAROUND_9641_HPP

#include "Kokkos_Core.hpp"

#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
#    include "plog/Log.h"
#endif

namespace Tests::Utils {

/**
 * @brief Workaround for https://github.com/kokkos/kokkos/issues/9641.
 *
 * If @c Exec can access the @c ViewType memory space, but is not the default
 * execution space of the @c ViewType memory space, make it an unmanaged view on
 * the @c Exec memory space.
 *
 * Otherwise, return the view as-is.
 *
 * An example is when @c Kokkos::OpenMP is used with a reduction target in @c Kokkos::CudaUVMSpace.
 */
template <Kokkos::ExecutionSpace Exec, typename ViewType>
requires Kokkos::is_view_v<std::remove_cvref_t<ViewType>>
auto reduction_workaround_9641(ViewType&& view) {
    using traits_t = typename std::remove_cvref_t<ViewType>::traits;

    if constexpr (
        Kokkos::SpaceAccessibility<Exec, typename std::remove_cvref_t<ViewType>::memory_space>::accessible
        && !std::same_as<Exec, typename std::remove_cvref_t<ViewType>::memory_space::execution_space>) {
        static_assert(
            Kokkos::Impl::is_default_memory_trait<typename traits_t::memory_traits>::value,
            "Erasing a non-default memory trait is hazardous.");

        using result_t = Kokkos::View<
            typename traits_t::data_type,
            typename traits_t::array_layout,
            typename Exec::memory_space,
            typename traits_t::hooks_policy,
            Kokkos::MemoryTraits<Kokkos::Unmanaged>
        >;

        return result_t(view.data(), view.layout());
    } else {
        return std::forward<ViewType>(view);
    }
}

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_REDUCTION_WORKAROUND_9641_HPP
