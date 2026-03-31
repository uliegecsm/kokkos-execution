#ifndef KOKKOS_EXECUTION_TESTS_UTILS_CHECK_CONTINUES_ON_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_CHECK_CONTINUES_ON_HPP

#include <concepts>

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/impl/sender_introspection.hpp"

#include "tests/utils/stdexec.hpp"

namespace Tests::Utils {

/**
 * @brief Check how the scheduler customizes @c stdexec::continues_on.
 *
 * Use this function to ensure your own scheduler is properly customizing everything that's needed.
 */
template <stdexec::scheduler Schd>
consteval bool check_continues_on() {
    using sndr_t = decltype(stdexec::just() | stdexec::continues_on(std::declval<Schd>()));

    //! Check the complete "demangled" sender type.
    static_assert(std::same_as<
                  stdexec::__demangle_t<sndr_t>,
                  basic_sender_t<
                      stdexec::continues_on_t,
                      Schd,
                      basic_sender_t<
                          stdexec::schedule_from_t,
                          stdexec::__,
                          basic_sender_t<stdexec::just_t, stdexec::__tup::__tuple<>>
                      >
                  >
    >);

    //! Diagnose any issue that could make the resulting sender invalid.
    if constexpr (stdexec::sender_in<sndr_t>) {
        stdexec::__diagnose_sender_concept_failure<sndr_t>();
    } else {
        stdexec::__diagnose_sender_concept_failure<sndr_t, stdexec::env<>>();
    }

    //! Check the completing domain;
    static_assert(std::same_as<stdexec::__domain_of_t<stdexec::env_of_t<sndr_t>>, stdexec::default_domain>);
    static_assert(std::same_as<
                  stdexec::__detail::__completing_domain_t<stdexec::set_value_t, sndr_t>,
                  std::invoke_result_t<stdexec::get_completion_domain_t<stdexec::set_value_t>, Schd>
    >);

    //! It must advertise a valid completion scheduler.
    if constexpr (
        stdexec::__minvocable_q<Kokkos::Execution::Impl::completion_scheduler_of_t, stdexec::set_value_t, sndr_t>) {
        static_assert(
            std::same_as<Kokkos::Execution::Impl::completion_scheduler_of_t<stdexec::set_value_t, sndr_t>, Schd>);
        //! Handle the case of dependent senders.
    } else {
        static_assert(stdexec::dependent_sender<sndr_t>);
        static_assert(std::same_as<
                      Kokkos::Execution::Impl::completion_scheduler_of_t<stdexec::set_value_t, sndr_t, stdexec::env<>>,
                      Schd
        >);
    }

    return true;
}

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_CHECK_CONTINUES_ON_HPP
