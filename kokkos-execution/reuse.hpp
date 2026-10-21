#ifndef KOKKOS_EXECUTION_PARALLEL_FOR_HPP
#define KOKKOS_EXECUTION_PARALLEL_FOR_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/completion_signatures.hpp"
#include "kokkos-execution/impl/env.hpp"
#include "kokkos-execution/impl/iteration_space.hpp"

namespace Kokkos::Execution {

namespace Impl {

template <stdexec::sender Sndr, typename Resource, stdexec::sender InnerSndr>
struct ReuseSender;

template <typename Resource, stdexec::sender InnerSndr>
struct ReuseData;

} // namespace Impl

/**
 * @brief Custom sender adaptor that allows explicit resource reuse.
 *
 * @code
 * Kokkos::Execution::reuse(resource, inner_sndr)
 * @endcode
 *
 * @todo Work out a paper-like description of the adaptor.
 * @todo There may be some constrains to apply on the resource (make it a concept).
 */
struct reuse_t {
    template <typename Resource, stdexec::sender InnerSndr>
    constexpr auto operator()(Resource&& resource, InnerSndr&& inner_sndr) const
        noexcept(stdexec::__nothrow_decay_copyable<Resource, InnerSndr>) {
        return stdexec::__closure(*this, std::forward<Resource>(resource), std::forward<InnerSndr>(inner_sndr));
    }

    template <stdexec::sender Sndr, typename Resource, stdexec::sender InnerSndr>
    constexpr auto operator()(Sndr&& sndr, Resource&& resource, InnerSndr&& inner_sndr) const
        noexcept(stdexec::__nothrow_decay_copyable<Sndr, Resource, InnerSndr>)
            -> Impl::ReuseSender<Sndr, std::remove_cvref_t<Resource>, InnerSndr> {
        return {
            {reuse_t{},
             Impl::ReuseData{std::forward<Resource>(resource), std::forward<InnerSndr>(inner_sndr)},
             std::forward<Sndr>(sndr)}
        };
    }
};

namespace Impl {

template <typename Resource, stdexec::sender InnerSndr>
struct ReuseData {
    Resource resource;
    InnerSndr inner_sender; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

//! Deduction guide to store the resource by-value.
template <typename Resource, stdexec::sender InnerSndr>
ReuseData(Resource, InnerSndr&&) -> ReuseData<Resource, InnerSndr>;

template <stdexec::sender Sndr, typename Resource, stdexec::sender InnerSndr>
struct ReuseSender : stdexec::__tuple<reuse_t, ReuseData<Resource, InnerSndr>, Sndr> {
    using sender_concept = stdexec::sender_tag;

    /// @name Inspired by https://github.com/NVIDIA/stdexec/blob/d76067bd3e765f1718ea1ff886f8c68e63b91a6f/include/stdexec/__detail/__basic_sender.hpp#L321-L
    ///@{
    using __data_t = ReuseData<Resource, InnerSndr>; // NOLINT(bugprone-reserved-identifier)
    using __children_t = stdexec::__mlist<Sndr>;     // NOLINT(bugprone-reserved-identifier)
    ///@}

    using base_t = stdexec::__tuple<reuse_t, __data_t, Sndr>;

    // not sure about the unconditional addition of error... this should probably become a dependent sender ?
    KOKKOS_EXECUTION_COMPL_SIGS_ADD(ReuseSender, Sndr, stdexec::set_error_t(std::exception_ptr))

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

inline constexpr reuse_t reuse{};

} // namespace Kokkos::Execution

#endif // KOKKOS_EXECUTION_PARALLEL_FOR_HPP
