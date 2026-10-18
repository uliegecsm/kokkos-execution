#ifndef KOKKOS_EXECUTION_TESTS_UTILS_PROBE_OPSTATE_START_THREAD_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_PROBE_OPSTATE_START_THREAD_HPP

#include <thread>

#include "kokkos-execution/stdexec.hpp"

namespace Tests::Utils {

//! Retrieve the thread ID on which the @ref OpState is started.
struct ProbeOpStateStartThreadSender {
    using sender_concept = stdexec::sender_tag;

    using completion_signatures = stdexec::completion_signatures<stdexec::set_value_t()>;

    std::thread::id* tid;

    template <typename Rcvr>
    struct OpState {
        using operation_state_concept = stdexec::operation_state_tag;

        std::thread::id* tid;
        Rcvr rcvr;

        void start() & noexcept {
            *tid = std::this_thread::get_id();
            stdexec::set_value(std::move(rcvr));
        }
    };

    template <stdexec::receiver_of<completion_signatures> Rcvr>
    constexpr auto connect(Rcvr&& rcvr) const noexcept -> OpState<Rcvr> {
        return {tid, std::forward<Rcvr>(rcvr)};
    }
};

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_PROBE_OPSTATE_START_THREAD_HPP
