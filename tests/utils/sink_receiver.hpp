#ifndef KOKKOS_EXECUTION_TESTS_UTILS_SINK_RECEIVER_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_SINK_RECEIVER_HPP

#include "kokkos-execution/stdexec.hpp"

namespace Tests::Utils {

//! A receiver that can handle all completions and does nothing with them.
struct SinkReceiver {
    using receiver_concept = stdexec::receiver_tag;

    void set_value(auto&&...) && noexcept {
    }
    void set_error(auto&&) && noexcept {
    }
    void set_stopped() && noexcept {
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> stdexec::env<> {
        return {};
    }
};

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_SINK_RECEIVER_HPP
