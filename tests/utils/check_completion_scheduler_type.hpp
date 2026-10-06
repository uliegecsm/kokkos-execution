#ifndef KOKKOS_EXECUTION_TESTS_UTILS_CHECK_COMPLETION_SCHEDULER_TYPE_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_CHECK_COMPLETION_SCHEDULER_TYPE_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/completion_signatures.hpp"
#include "kokkos-execution/impl/env.hpp"
#include "kokkos-execution/impl/sender_introspection.hpp"
#include "kokkos-execution/impl/type_traits.hpp"

#include "tests/utils/stdexec.hpp"

/**
 * @file
 *
 * Sender that statically asserts the completion scheduler type at connect time.
 */

namespace Tests::Utils {

template <stdexec::sender Sndr, typename Tag, stdexec::scheduler Schd>
struct CheckCompletionSchedulerTypeSender;

template <typename Tag, stdexec::scheduler Schd>
struct check_completion_scheduler_type_t {
    [[nodiscard]]
    constexpr auto operator()() const noexcept {
        return stdexec::__closure(*this);
    }

    template <stdexec::sender Sndr>
    [[nodiscard]]
    constexpr auto operator()(Sndr&& sndr) const {
        return CheckCompletionSchedulerTypeSender<Sndr, Tag, Schd>{std::forward<Sndr>(sndr)};
    }
};

template <stdexec::sender Sndr, typename Tag, stdexec::scheduler Schd>
struct CheckCompletionSchedulerTypeSender {
    using sender_concept = stdexec::sender_tag;

    Sndr sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)

    KOKKOS_EXECUTION_COMPL_SIGS_KEEP(CheckCompletionSchedulerTypeSender, Sndr)

    template <stdexec::__decays_to<CheckCompletionSchedulerTypeSender> Self, stdexec::receiver Rcvr>
    [[nodiscard]]
    constexpr STDEXEC_EXPLICIT_THIS_BEGIN(
        auto connect)(this Self&& self, Rcvr rcvr) // NOLINT(cppcoreguidelines-missing-std-forward)
        noexcept(stdexec::__nothrow_connectable<KOKKOS_EXECUTION_IMPL_MEMBER_CVREF_T(Self, sndr), Rcvr&&>)
            -> stdexec::connect_result_t<KOKKOS_EXECUTION_IMPL_MEMBER_CVREF_T(Self, sndr), Rcvr&&> {
        static_assert(check_completion_scheduler_type<Rcvr>());
        return stdexec::connect(std::forward<Self>(self).sndr, std::move(rcvr));
    }
    STDEXEC_EXPLICIT_THIS_END(connect)

    template <stdexec::receiver Rcvr>
    static consteval bool check_completion_scheduler_type() {
        if constexpr (Tests::Utils::has_completion_scheduler_for<Sndr, Tag, stdexec::env_of_t<Rcvr>>) {
            using schd_t = Kokkos::Execution::Impl::completion_scheduler_of_t<Tag, Sndr, stdexec::env_of_t<Rcvr>>;
            static_assert(
                std::same_as<std::remove_cvref_t<schd_t>, Schd>,
                "Scheduler type mismatch: completion scheduler type doesn't match expected type.");
            return true;
        } else {
            static_assert(sizeof(Rcvr) == 0, "No completion scheduler found.");
            return false;
        }
    }

    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(Sndr, sndr)
};

template <typename Tag, stdexec::scheduler Schd>
inline constexpr check_completion_scheduler_type_t<Tag, Schd> check_completion_scheduler_type{};

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_CHECK_COMPLETION_SCHEDULER_TYPE_HPP
