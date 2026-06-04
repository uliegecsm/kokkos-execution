#ifndef KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_INCREMENT_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_INCREMENT_HPP

#include <type_traits>

#include "kokkos-execution/stdexec.hpp"

#include "Kokkos_Core.hpp"

#include "kokkos-utils/concepts/View.hpp"

#include "tests/utils/atomic.hpp"

namespace Tests::Utils::Functors {

//! Increment @ref data.
template <
    Kokkos::utils::concepts::ViewOfRank<0> ViewType,
    bool MayThrow = true,
    bool Atomic = false,
    typename MemoryScope = desul::MemoryScopeDevice
>
struct Increment {
    typename ViewType::non_const_type data;

    KOKKOS_FUNCTION
    void operator()() const noexcept(!MayThrow) requires(Atomic == false)
    {
        ++data();
    }

    KOKKOS_FUNCTION
    void operator()() const noexcept(!MayThrow) requires(Atomic == true)
    {
        Tests::Utils::atomic_fetch_add<MemoryScope, desul::MemoryOrderRelaxed, typename ViewType::non_const_value_type>(
            data.data(), 1);
    }
};

template <
    Kokkos::utils::concepts::ViewOfRank<0> ViewType,
    bool MayThrow = true,
    typename MemoryScope = desul::MemoryScopeDevice
>
struct FetchIncrement {
    typename ViewType::non_const_type counter, value;

    KOKKOS_FUNCTION
    void operator()() const noexcept(!MayThrow) {
        value() = Tests::Utils::atomic_fetch_add<
            MemoryScope,
            desul::MemoryOrderRelaxed,
            typename ViewType::non_const_value_type
        >(counter.data(), 1);
    }
};

//! Add a @c then using @ref Tests::Utils::Functors::Increment that may throw. // NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define THEN_INCREMENT(_data_)                                                                                         \
    stdexec::then(Tests::Utils::Functors::Increment<std::remove_cvref_t<decltype(_data_)>, true, false>{.data = _data_})

//! Same as @ref THEN_INCREMENT, using @ref Tests::Utils::atomic_fetch_add. // NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define THEN_INCREMENT_ATOMIC(_scope_, _data_)                                                                         \
    stdexec::then(                                                                                                     \
        Tests::Utils::Functors::Increment<                                                                             \
            std::remove_cvref_t<decltype(_data_)>,                                                                     \
            true,                                                                                                      \
            true,                                                                                                      \
            desul::MemoryScope##_scope_                                                                                \
        >{.data = _data_})

} // namespace Tests::Utils::Functors

#endif // KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_INCREMENT_HPP
