#ifndef KOKKOS_EXECUTION_IMPL_SENDER_CONCEPTS_HPP
#define KOKKOS_EXECUTION_IMPL_SENDER_CONCEPTS_HPP

#include "kokkos-execution/parallel_for.hpp"
#include "kokkos-execution/stdexec.hpp"

namespace Kokkos::Execution::Impl {

//! Concept that constrains the type of a sender that dispatches a functor for execution.
template <typename Sndr>
concept dispatching_sender = stdexec::sender<Sndr> && requires {
    typename stdexec::tag_of_t<Sndr>;
    requires stdexec::__one_of<
        stdexec::tag_of_t<Sndr>,
        stdexec::bulk_t,
        stdexec::then_t,
        Kokkos::Execution::parallel_for_t
    >;
};

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_SENDER_CONCEPTS_HPP
