#ifndef KOKKOS_EXECUTION_IMPL_ITERATION_SPACE_HPP
#define KOKKOS_EXECUTION_IMPL_ITERATION_SPACE_HPP

#include <type_traits>

#include "kokkos-execution/stdexec.hpp"

#include "Kokkos_Macros.hpp"
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
import kokkos.execution.mdspan;
#else
#    include "Kokkos_Core.hpp"
#endif

namespace Kokkos::detail {
//! @c Kokkos::detail::is_constant_wrapper specialization for @c std::integral_constant.
template <typename T, T Value>
constexpr bool is_constant_wrapper<std::integral_constant<T, Value>> = true;
} // namespace Kokkos::detail

namespace Kokkos::Execution::Impl {

template <typename T>
concept extents = Kokkos::detail::impl_is_extents_v<T>;

template <typename T>
concept range_slice = Kokkos::detail::is_range_slice<T>::value;

//! Constrain a @c Kokkos::range_slice to have a compile-time known unit stride.
template <typename T>
concept range_slice_with_default_stride = range_slice<T>
                                       && Kokkos::detail::is_constant_wrapper<decltype(std::declval<T>().stride)>
                                       && (std::remove_cvref_t<decltype(std::declval<T>().stride)>::value == 1);

//! Constrain a @c Kokkos::range_slice to have compile-time known @c first and @c last, as well as a compile-time known unit stride.
template <typename T>
concept static_range_slice_with_default_stride = range_slice_with_default_stride<T>
                                              && Kokkos::detail::is_constant_wrapper<decltype(std::declval<T>().first)>
                                              && Kokkos::detail::is_constant_wrapper<decltype(std::declval<T>().last)>;

//! Concept that constrains types that may represent an iteration space.
template <typename T>
concept iteration_space = (extents<T> && requires { T::rank() == 1; }) || range_slice_with_default_stride<T>;

template <typename T>
concept execution_policy_or_iteration_space = Kokkos::ExecutionPolicy<T> || iteration_space<T>;

//! Make a valid iteration space from a work count.
template <std::integral T>
constexpr auto make_iter_space(const T work_count) noexcept -> Kokkos::dims<1, T> {
    return Kokkos::dims<1, T>{work_count};
}

template <extents Extents>
requires(Extents::rank() == 1)
constexpr decltype(auto) begin(const Extents&) noexcept {
    return 0;
}

template <extents Extents>
requires(Extents::rank() == 1)
constexpr decltype(auto) end(const Extents& range) noexcept {
    return range.extent(0);
}

template <range_slice RangeSlice>
constexpr decltype(auto) begin(const RangeSlice& rgsl) noexcept {
    return rgsl.first;
}

template <range_slice RangeSlice>
constexpr decltype(auto) end(const RangeSlice& rgsl) noexcept {
    return rgsl.last;
}

template <Kokkos::ExecutionPolicy ExecPolicy>
requires stdexec::__is_instance_of<ExecPolicy, Kokkos::RangePolicy>
constexpr decltype(auto) begin(const ExecPolicy& rgpl) noexcept {
    return rgpl.begin();
}

template <Kokkos::ExecutionPolicy ExecPolicy>
requires stdexec::__is_instance_of<ExecPolicy, Kokkos::RangePolicy>
constexpr decltype(auto) end(const ExecPolicy& rgpl) noexcept {
    return rgpl.end();
}

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_ITERATION_SPACE_HPP
