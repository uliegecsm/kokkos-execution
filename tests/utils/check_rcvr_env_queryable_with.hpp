#ifndef KOKKOS_EXECUTION_TESTS_UTILS_CHECK_RCVR_ENV_QUERYABLE_WITH_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_CHECK_RCVR_ENV_QUERYABLE_WITH_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/completion_signatures.hpp"
#include "kokkos-execution/impl/env.hpp"
#include "kokkos-execution/impl/type_traits.hpp"

#include "tests/utils/stdexec.hpp"

/**
 * @file
 *
 * Check the environment of the receiver is queryable for a given set of queries.
 */

namespace Tests::Utils {

template <bool IsQueryable, stdexec::sender Sndr, typename... Queries>
struct CheckRcvrEnvQueryableWithSender;

template <bool IsQueryable, typename... Queries>
struct check_rcvr_env_queryable_with_t {
    [[nodiscard]]
    constexpr auto operator()() const noexcept {
        return stdexec::__closure(*this);
    }

    template <stdexec::sender Sndr>
    [[nodiscard]]
    constexpr auto operator()(Sndr&& sndr) const {
        return CheckRcvrEnvQueryableWithSender<IsQueryable, Sndr, Queries...>{std::forward<Sndr>(sndr)};
    }
};

template <bool IsQueryable, stdexec::sender Sndr, typename... Queries>
struct CheckRcvrEnvQueryableWithSender {
    using sender_concept = stdexec::sender_tag;

    Sndr sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)

    KOKKOS_EXECUTION_COMPL_SIGS_KEEP(CheckRcvrEnvQueryableWithSender)

    template <stdexec::__decays_to<CheckRcvrEnvQueryableWithSender> Self, stdexec::receiver Rcvr>
    [[nodiscard]]
    constexpr STDEXEC_EXPLICIT_THIS_BEGIN(
        auto connect)(this Self&& self, Rcvr rcvr) // NOLINT(cppcoreguidelines-missing-std-forward)
        noexcept(stdexec::__nothrow_connectable<KOKKOS_EXECUTION_IMPL_MEMBER_CVREF_T(Self, sndr), Rcvr&&>)
            -> stdexec::connect_result_t<KOKKOS_EXECUTION_IMPL_MEMBER_CVREF_T(Self, sndr), Rcvr&&> {
        static_assert(check_rcvr_env<Rcvr>());
        return stdexec::connect(std::forward<Self>(self).sndr, std::move(rcvr));
    }
    STDEXEC_EXPLICIT_THIS_END(connect)

    template <stdexec::receiver Rcvr>
    static consteval bool check_rcvr_env() {
        if constexpr (IsQueryable) {
            static_assert(
                (stdexec::__queryable_with<stdexec::env_of_t<Rcvr>, Queries> && ...),
                "The receiver environment is not queryable with at least one query from the given set of queries.");
        } else {
            static_assert(
                ((!stdexec::__queryable_with<stdexec::env_of_t<Rcvr>, Queries>) && ...),
                "The receiver environment is queryable with at least one query from the given set of queries.");
        }
        return true;
    }

    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(Sndr, sndr)
};

template <typename... Queries>
inline constexpr check_rcvr_env_queryable_with_t<true, Queries...> check_rcvr_env_queryable_with{};

template <typename... Queries>
inline constexpr check_rcvr_env_queryable_with_t<false, Queries...> check_rcvr_env_not_queryable_with{};

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_CHECK_RCVR_ENV_QUERYABLE_WITH_HPP
