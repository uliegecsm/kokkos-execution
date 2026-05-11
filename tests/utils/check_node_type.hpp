#ifndef KOKKOS_EXECUTION_TESTS_UTILS_CHECK_NODE_TYPE_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_CHECK_NODE_TYPE_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/graph/get_node.hpp"

/**
 * @file
 *
 * Check the @c Kokkos::Graph node type of the preceding operation state.
 */

namespace Tests::Utils {

template <typename T, stdexec::sender Sndr>
struct CheckNodeTypeSender;

template <typename T>
struct check_node_type_t {
    [[nodiscard]]
    constexpr auto operator()() const noexcept {
        return stdexec::__closure(*this);
    }

    template <stdexec::sender Sndr>
    [[nodiscard]]
    constexpr auto operator()(Sndr&& sndr) const {
        return CheckNodeTypeSender<T, Sndr>{std::forward<Sndr>(sndr)};
    }
};

template <typename T, stdexec::sender Sndr>
struct CheckNodeTypeSender {
    using sender_concept = stdexec::sender_tag;

    Sndr sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)

    KOKKOS_EXECUTION_COMPL_SIGS_KEEP(CheckNodeTypeSender, Sndr)

    template <stdexec::__decays_to<CheckNodeTypeSender> Self, stdexec::receiver Rcvr>
    [[nodiscard]]
    constexpr STDEXEC_EXPLICIT_THIS_BEGIN(
        auto connect)(this Self&& self, Rcvr rcvr) // NOLINT(cppcoreguidelines-missing-std-forward)
        noexcept(stdexec::__nothrow_connectable<KOKKOS_EXECUTION_IMPL_MEMBER_CVREF_T(Self, sndr), Rcvr&&>)
            -> stdexec::connect_result_t<KOKKOS_EXECUTION_IMPL_MEMBER_CVREF_T(Self, sndr), Rcvr&&> {
        using op_state_t = stdexec::connect_result_t<KOKKOS_EXECUTION_IMPL_MEMBER_CVREF_T(Self, sndr), Rcvr&&>;
        using node_t = stdexec::__query_result_t<op_state_t, Kokkos::Execution::GraphImpl::get_node_t>;
        static_assert(std::same_as<std::remove_cvref_t<node_t>, std::remove_cvref_t<T>>);
        return stdexec::connect(std::forward<Self>(self).sndr, std::move(rcvr));
    }
    STDEXEC_EXPLICIT_THIS_END(connect)

    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(Sndr, sndr)
};

//! The provided type @c T is compared without cv type qualifiers during @c connect.
template <typename T>
inline constexpr check_node_type_t<T> check_node_type{};

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_CHECK_NODE_TYPE_HPP
