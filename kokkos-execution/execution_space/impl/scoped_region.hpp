#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_IMPL_SCOPED_REGION_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_IMPL_SCOPED_REGION_HPP

#include <format>

#include "kokkos-execution/utils/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsuggest-override")
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "impl/Kokkos_Profiling.hpp"

#include "kokkos-execution/execution_space/Context_fwd.hpp"
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

namespace Kokkos::Execution::execution_space::impl {
//! Kind of region action.
enum class Kind : std::uint8_t {
    PUSH,
    POP
};

template <Kind kind, stdexec::receiver Rcvr, stdexec::__is_instance_of<Scheduler> Schd>
struct RegionReceiver {
    using receiver_concept = stdexec::receiver_t;

    Rcvr rcvr;
    std::string name;
    Schd schd;

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
    using sender_concept = stdexec::sender_t;

    KOKKOS_EXECUTION_COMPL_SIGS_KEEP(RegionSender)

    template <stdexec::receiver Rcvr>
    stdexec::operation_state auto connect(Rcvr rcvr) && noexcept(std::is_nothrow_move_constructible_v<Rcvr>) {
        auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr));

        using recv_t = RegionReceiver<kind, Rcvr, std::remove_cvref_t<decltype(schd)>>;

        return stdexec::connect(
            std::move(sndr), recv_t{.rcvr = std::move(rcvr), .name = std::move(name), .schd = std::move(schd)});
    }

    Sndr sndr;
    std::string name{};

    KOKKOS_EXECUTION_FORWARDING_GET_ENV(Sndr, sndr)
};

struct Push {
    template <stdexec::sender Sndr, typename T>
    auto operator()(Sndr&& sndr, T&& name) const noexcept -> RegionSender<Kind::PUSH, Sndr> {
        return RegionSender<Kind::PUSH, Sndr>{.sndr = std::forward<Sndr>(sndr), .name = std::forward<T>(name)};
    }

    template <typename T>
    auto operator()(T&& name) const noexcept {
        return stdexec::__closure{*this, std::forward<T>(name)};
    }
};

struct Pop {
    template <stdexec::sender Sndr>
    auto operator()(Sndr&& sndr) const noexcept -> RegionSender<Kind::POP, Sndr> {
        return RegionSender<Kind::POP, Sndr>{.sndr = std::forward<Sndr>(sndr)};
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

} // namespace Kokkos::Execution::execution_space::impl

namespace Kokkos::Profiling {
inline constexpr Kokkos::Execution::execution_space::impl::Push push{};
inline constexpr Kokkos::Execution::execution_space::impl::Pop pop{};
inline constexpr Kokkos::Execution::execution_space::impl::ScopedRegion scoped_region{};
} // namespace Kokkos::Profiling

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_IMPL_SCOPED_REGION_HPP
