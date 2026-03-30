#ifndef KOKKOS_EXECUTION_TESTS_UTILS_CHECK_SCHEDULER_TYPE_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_CHECK_SCHEDULER_TYPE_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/completion_signatures.hpp"
#include "kokkos-execution/impl/env.hpp"

#include "tests/utils/stdexec.hpp"

/**
 * @file
 *
 * Sender that statically asserts the scheduler type at connect time.
 */

namespace Tests::Utils {

template <stdexec::sender Sndr, typename Tag, stdexec::scheduler Schd>
struct CheckSchedulerTypeSender;

template <typename Tag, stdexec::scheduler Schd>
struct check_scheduler_type_t {
    [[nodiscard]]
    constexpr auto operator()() const noexcept {
        return stdexec::__closure(*this);
    }

    template <stdexec::sender Sndr>
    [[nodiscard]]
    constexpr auto operator()(Sndr&& sndr) const {
        return CheckSchedulerTypeSender<Sndr, Tag, Schd>{std::forward<Sndr>(sndr)};
    }
};

template <stdexec::sender Sndr, typename Tag, stdexec::scheduler Schd>
struct CheckSchedulerTypeSender {
    using sender_concept = stdexec::sender_t;

    Sndr sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)

    KOKKOS_EXECUTION_COMPL_SIGS_KEEP(CheckSchedulerTypeSender)

    template <stdexec::receiver Rcvr>
    constexpr auto connect(Rcvr rcvr) && noexcept(stdexec::__nothrow_connectable<Sndr&&, Rcvr&&>) {
        static_assert(check_scheduler_type<Rcvr>());
        return stdexec::connect(std::forward<Sndr>(sndr), std::move(rcvr));
    }

    template <stdexec::receiver Rcvr>
    constexpr auto connect(Rcvr rcvr) const & noexcept(stdexec::__nothrow_connectable<const Sndr&, Rcvr&&>) {
        static_assert(check_scheduler_type<Rcvr>());
        return stdexec::connect(sndr, std::move(rcvr));
    }

    template <stdexec::receiver Rcvr>
    static consteval bool check_scheduler_type() {
        /// First, try to get the completion scheduler from the sender environment.
        if constexpr (Tests::Utils::has_completion_scheduler_for<Sndr, Tag, stdexec::env_of_t<Rcvr>>) {
            using schd_t = stdexec::__completion_scheduler_of_t<Tag, Sndr, stdexec::env_of_t<Rcvr>>;
            static_assert(
                std::same_as<std::remove_cvref_t<schd_t>, Schd>,
                "Scheduler type mismatch: completion scheduler doesn't match expected type.");
            return true;
        }
        /// Fallback on the receiver environment.
        else if constexpr (stdexec::__queryable_with<stdexec::env_of_t<Rcvr>, stdexec::get_scheduler_t>) {
            using schd_t = stdexec::__query_result_t<stdexec::env_of_t<Rcvr>, stdexec::get_scheduler_t>;
            static_assert(
                std::same_as<std::remove_cvref_t<schd_t>, Schd>,
                "Scheduler type mismatch: receiver scheduler doesn't match expected type.");
            return true;
        } else {
            static_assert(sizeof(Rcvr) == 0, "No scheduler found.");
            return false;
        }
    }

    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(Sndr, sndr)
};

template <typename Tag, stdexec::scheduler Schd>
inline constexpr check_scheduler_type_t<Tag, Schd> check_scheduler_type{};

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_CHECK_SCHEDULER_TYPE_HPP
