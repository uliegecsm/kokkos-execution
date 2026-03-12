#ifndef KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_INCREMENT_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_INCREMENT_HPP

#include <type_traits>

#include "kokkos-execution/stdexec.hpp"

#include "Kokkos_Core.hpp"

#include "kokkos-utils/concepts/View.hpp"

namespace Tests::Utils::Functors {

//! Increment @ref data.
template <Kokkos::utils::concepts::ViewOfRank<0> ViewType, bool MayThrow = true>
struct Increment {
    typename ViewType::non_const_type data;

    KOKKOS_FUNCTION
    void operator()() const noexcept(!MayThrow) {
        ++data();
    }
};

//! Add a @c then using @ref Tests::Utils::Functors::Increment that may throw. // NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define THEN_INCREMENT(_data_)                                                                                         \
    stdexec::then(Tests::Utils::Functors::Increment<std::remove_cvref_t<decltype(_data_)>, true>{.data = _data_})

template <Kokkos::utils::concepts::View ViewType>
using as_atomic_view_t = Kokkos::View<
    typename ViewType::traits::data_type,
    typename ViewType::traits::array_layout,
    typename ViewType::traits::device_type,
    typename ViewType::traits::hooks_policy,
    Kokkos::MemoryTraits<ViewType::traits::memory_traits::impl_value | Kokkos::Atomic>
>;

//! Same as @ref THEN_INCREMENT with atomic memory traits. // NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define THEN_INCREMENT_ATOMIC(_data_)                                                                                  \
    stdexec::then(                                                                                                     \
        Tests::Utils::Functors::Increment<                                                                             \
            Tests::Utils::Functors::as_atomic_view_t<std::remove_cvref_t<decltype(_data_)>>,                           \
            true                                                                                                       \
        >{.data = _data_})

} // namespace Tests::Utils::Functors

#endif // KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_INCREMENT_HPP
