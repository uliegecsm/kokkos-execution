#ifndef KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_LABELED_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_LABELED_HPP

#include "kokkos-execution/stdexec.hpp"

#include "Kokkos_Core.hpp"

namespace Tests::Utils::Functors {

template <char ID>
struct Labeled {
    template <typename... Args>
    KOKKOS_FUNCTION void operator()(Args...) const noexcept {
    }
};

//! Add a @c then using @ref Tests::Utils::Functors::Labeled. // NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define THEN_LABELED(_id_) stdexec::then(Tests::Utils::Functors::Labeled<_id_>{})

//! Add a @ref Kokkos::Execution::parallel_for using @ref Tests::Utils::Functors::Labeled. // NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define THEN_LABELED_PFOR(_exec_, _id_)                                                                                \
    Kokkos::Execution::parallel_for(#_id_, Kokkos::RangePolicy<_exec_>(0, 1), Tests::Utils::Functors::Labeled<_id_>{})

} // namespace Tests::Utils::Functors

#endif // KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_LABELED_HPP
