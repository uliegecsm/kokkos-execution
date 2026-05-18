#ifndef KOKKOS_EXECUTION_TESTS_UTILS_CHECK_RCVR_ENV_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_CHECK_RCVR_ENV_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/completion_signatures.hpp"
#include "kokkos-execution/impl/type_traits.hpp"

namespace Tests::Utils {

template <typename ExpectedEnv, typename Sndr>
struct CheckRcvrEnvSender {
    using sender_concept = stdexec::sender_tag;

    Sndr sndr;

    KOKKOS_EXECUTION_COMPL_SIGS_KEEP(CheckRcvrEnvSender)

    template <stdexec::__decays_to<CheckRcvrEnvSender> Self, stdexec::receiver Rcvr>
    [[nodiscard]]
    constexpr STDEXEC_EXPLICIT_THIS_BEGIN(
        auto connect)(this Self&& self, Rcvr rcvr) // NOLINT(cppcoreguidelines-missing-std-forward)
        noexcept(stdexec::__nothrow_connectable<KOKKOS_EXECUTION_IMPL_MEMBER_CVREF_T(Self, sndr), Rcvr&&>)
            -> stdexec::connect_result_t<KOKKOS_EXECUTION_IMPL_MEMBER_CVREF_T(Self, sndr), Rcvr&&> {
        static_assert(std::same_as<ExpectedEnv, stdexec::env_of_t<Rcvr>>);
        return stdexec::connect(std::forward<Self>(self).sndr, std::move(rcvr));
    }
    STDEXEC_EXPLICIT_THIS_END(connect)

    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(Sndr, sndr)
};

//! Check that the receiver environment is of type @p ExpectedEnv.
template <typename ExpectedEnv>
struct check_rcvr_env_t {
    [[nodiscard]]
    constexpr auto operator()() const noexcept {
        return stdexec::__closure(*this);
    }

    template <stdexec::sender Sndr>
    [[nodiscard]]
    constexpr auto operator()(Sndr&& sndr) const {
        return CheckRcvrEnvSender<ExpectedEnv, Sndr>{std::forward<Sndr>(sndr)};
    }
};

template <typename ExpectedEnv>
inline constexpr check_rcvr_env_t<ExpectedEnv> check_rcvr_env{};

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_CHECK_RCVR_ENV_HPP
