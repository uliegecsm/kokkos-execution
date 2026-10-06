#ifndef KOKKOS_EXECUTION_TESTS_UTILS_CHECK_NODE_TYPE_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_CHECK_NODE_TYPE_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/graph/get_node.hpp"
#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/completion_signatures.hpp"

/**
 * @file
 *
 * Check the underlying @c Kokkos::Graph node type.
 */

namespace Tests::Utils {

template <typename NodeType, stdexec::sender Sndr>
struct CheckNodeTypeSender;

template <typename NodeType>
struct check_node_type_t {
    [[nodiscard]]
    constexpr auto operator()() const noexcept {
        return stdexec::__closure(*this);
    }

    template <stdexec::sender Sndr>
    [[nodiscard]]
    constexpr auto operator()(Sndr&& sndr) const {
        return CheckNodeTypeSender<NodeType, Sndr>{std::forward<Sndr>(sndr)};
    }
};

template <typename NodeType, stdexec::sender Sndr>
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
        static_assert(stdexec::__queryable_with<op_state_t, Kokkos::Execution::GraphImpl::get_node_t>);
        using node_t = stdexec::__query_result_t<op_state_t, Kokkos::Execution::GraphImpl::get_node_t>;
        static_assert(std::same_as<std::remove_cvref_t<node_t>, std::remove_cvref_t<NodeType>>);
        return stdexec::connect(std::forward<Self>(self).sndr, std::move(rcvr));
    }
    STDEXEC_EXPLICIT_THIS_END(connect)

    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(Sndr, sndr)
};

//! The provided type @c NodeType is compared without cv type qualifiers during @c connect.
template <typename NodeType>
inline constexpr check_node_type_t<NodeType> check_node_type{};

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_CHECK_NODE_TYPE_HPP
