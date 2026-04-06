#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_SCOPED_REGION_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_SCOPED_REGION_HPP

#include <format>

#include "kokkos-execution/stdexec.hpp"

#include "impl/Kokkos_Profiling.hpp"

#include "kokkos-execution/execution_space/execution_space_fwd.hpp"

#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/completion_signatures.hpp"
#include "kokkos-execution/impl/env.hpp"

/**
 * @file
 *
 * This file provides the implementation of algorithms related to @c Kokkos::Profiling push/pop regions.
 *
 * References:
 *  - https://github.com/NVIDIA/stdexec/blob/f7308ea245b896a76c6fd9a58a097ae23579e489/include/nvexec/nvtx.cuh
 */

namespace Kokkos::Execution::ExecutionSpaceImpl {

//! Kind of region action.
enum class Kind : std::uint8_t {
    PUSH,
    POP
};

template <Kind kind, stdexec::__is_instance_of<Scheduler> Schd, stdexec::receiver Rcvr>
struct RegionReceiver {
    using receiver_concept = stdexec::receiver_tag;

    std::string name;
    Schd schd;
    Rcvr rcvr;

    template <typename Tag, typename... Args>
    void complete(Tag tag, Args&&... args) && noexcept {
        schd.state->exec.fence(
            std::format(
                "{}: {}",
                Kokkos::Impl::TypeInfo<typename Schd::execution_space>::name(),
                kind == Kind::PUSH ? "push" : "pop"));

        if constexpr (kind == Kind::PUSH) {
            Kokkos::Profiling::pushRegion(name);
        } else {
            Kokkos::Profiling::popRegion();
        }

        std::invoke(tag, std::move(rcvr), std::forward<Args>(args)...);
    }

    void set_value() && noexcept {
        std::move(*this).complete(stdexec::set_value);
    }

    template <typename Error>
    void set_error(Error&& err) && noexcept {
        std::move(*this).complete(stdexec::set_error, std::forward<Error>(err));
    }

    KOKKOS_EXECUTION_FORWARDING_GET_ENV(Rcvr, rcvr)
};

template <Kind kind, stdexec::sender Sndr>
struct RegionSender {
    using sender_concept = stdexec::sender_tag;

    KOKKOS_EXECUTION_COMPL_SIGS_KEEP(RegionSender)

    template <typename Rcvr>
    using schd_t = stdexec::__completion_scheduler_of_t<stdexec::set_value_t, Sndr, stdexec::env_of_t<Rcvr>>;

    template <typename Rcvr>
    using rcvr_t = RegionReceiver<kind, schd_t<Rcvr>, Rcvr>;

    template <stdexec::receiver Rcvr>
    stdexec::operation_state auto connect(Rcvr rcvr) && noexcept(
        std::is_nothrow_constructible_v<rcvr_t<Rcvr>, std::string&&, schd_t<Rcvr>&&, Rcvr&&>
        && stdexec::__nothrow_connectable<Sndr&&, rcvr_t<Rcvr>>) {
        auto schd =
            stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), stdexec::get_env(rcvr));

        return stdexec::connect(
            std::forward<Sndr>(sndr),
            rcvr_t<Rcvr>{.name = std::move(name), .schd = std::move(schd), .rcvr = std::move(rcvr)});
    }

    std::string name{};
    Sndr sndr;

    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(Sndr, sndr)
};

struct Push {
    template <stdexec::sender Sndr, typename T>
    auto operator()(Sndr&& sndr, T&& name) const noexcept -> RegionSender<Kind::PUSH, Sndr> {
        return {.name = std::forward<T>(name), .sndr = std::forward<Sndr>(sndr)};
    }

    template <typename T>
    auto operator()(T&& name) const noexcept {
        return stdexec::__closure{*this, std::forward<T>(name)};
    }
};

struct Pop {
    template <stdexec::sender Sndr>
    auto operator()(Sndr&& sndr) const noexcept -> RegionSender<Kind::POP, Sndr> {
        return {.sndr = std::forward<Sndr>(sndr)};
    }

    auto operator()() const noexcept {
        return stdexec::__closure{*this};
    }
};

//! Helper for @c Kokkos::Profiling::scoped_region.
struct ScopedRegion {
    template <stdexec::sender Sndr, typename T, stdexec::__sender_adaptor_closure Closure>
    auto operator()(Sndr&& sndr, T&& name, Closure&& closure) const noexcept {
        return std::forward<Sndr>(sndr) | Push{}(std::forward<T>(name)) | std::forward<Closure>(closure) | Pop{}();
    }

    template <typename T, stdexec::__sender_adaptor_closure Closure>
    auto operator()(T&& name, Closure&& closure) const noexcept {
        return stdexec::__closure{*this, std::forward<T>(name), std::forward<Closure>(closure)};
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

namespace Kokkos::Execution::Profiling {
inline constexpr Kokkos::Execution::ExecutionSpaceImpl::Push push{};
inline constexpr Kokkos::Execution::ExecutionSpaceImpl::Pop pop{};
inline constexpr Kokkos::Execution::ExecutionSpaceImpl::ScopedRegion scoped_region{};
} // namespace Kokkos::Execution::Profiling

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_SCOPED_REGION_HPP
