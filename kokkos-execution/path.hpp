#ifndef KOKKOS_EXECUTION_PATH_HPP
#define KOKKOS_EXECUTION_PATH_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/impl/macros.hpp"

namespace Kokkos::Execution {

namespace Impl {
template <typename Sndr, typename... Clsrs>
struct PathSender;
} // namespace Impl

/**
 * @brief Sender adaptor that gathers an ensemble of sender adaptor closures.
 *
 * A *path* is a sequence of vertices (the closures) linked by edges (the order in
 * which they are executed).
 *
 * @ref path_t allows a flat representation and construction
 * @code
 * const auto sndr = stdexec::just()
 *                 | Kokkos::Execution::path(
 *                       Kokkos::Execution::parallel_for(42, functor_a),
 *                       Kokkos::Execution::parallel_for(42, functor_b));
 * @endcode
 * which is semantically equivalent to
 * @code
 * const auto sndr = stdexec::just()
 *                 | Kokkos::Execution::parallel_for(42, functor_a)
 *                 | Kokkos::Execution::parallel_for(42, functor_b);
 * @endcode
 */
struct path_t {
    template <stdexec::__sender_adaptor_closure... Clsrs>
    constexpr auto
        operator()(Clsrs&&... clsrs) const STDEXEC_AUTO_RETURN(stdexec::__closure(*this, std::forward<Clsrs>(clsrs)...))

    template <stdexec::sender Sndr, stdexec::__sender_adaptor_closure... Clsrs>
    constexpr auto operator()(Sndr&& sndr, Clsrs&&... clsrs) const KOKKOS_EXECUTION_RETURNS_BRACED(
        (Impl::PathSender<Sndr, Clsrs...>),
        {path_t{}, stdexec::__mktuple(std::forward<Clsrs>(clsrs)...), std::forward<Sndr>(sndr)})
};

namespace Impl {
template <typename Sndr, typename... Clsrs>
struct PathSender : public stdexec::__tuple<path_t, stdexec::__tuple<Clsrs...>, Sndr> {
    using sender_concept = stdexec::sender_tag;

    /// @name Inspired by https://github.com/NVIDIA/stdexec/blob/d76067bd3e765f1718ea1ff886f8c68e63b91a6f/include/stdexec/__detail/__basic_sender.hpp#L321-L
    ///@{
    using __data_t = stdexec::__tuple<Clsrs...>; // NOLINT(bugprone-reserved-identifier)
    using __children_t = stdexec::__mlist<Sndr>; // NOLINT(bugprone-reserved-identifier)
    ///@}

    using base_t = stdexec::__tuple<path_t, __data_t, Sndr>;

    // completion signatures?

    template <stdexec::receiver Rcvr>
    constexpr auto connect(Rcvr) && = delete;

    template <stdexec::receiver Rcvr>
    constexpr auto connect(Rcvr) const & = delete;

    static constexpr size_t idx_sndr = 2;
    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(
        Sndr,
        stdexec::__get<idx_sndr>(static_cast<const base_t&>(*this)))
};

} // namespace Impl

inline constexpr path_t path{};

} // namespace Kokkos::Execution

#endif // KOKKOS_EXECUTION_PATH_HPP
