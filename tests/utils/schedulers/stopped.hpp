#ifndef KOKKOS_EXECUTION_TESTS_UTILS_SCHEDULERS_STOPPED_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_SCHEDULERS_STOPPED_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/impl/immovable.hpp"
#include "kokkos-execution/impl/macros.hpp"

#include "tests/utils/schedulers/attributes.hpp"

namespace Tests::Utils::Schedulers {
/**
 * @brief Scheduler that returns a sender that always completes on the stop channel.
 *
 * The @ref Stopped::Sender advertises both @c stdexec::set_stopped_t *and* @c stdexec::set_value_t, even though
 * @ref Stopped::OpState always starts on the @c stdexec::set_stopped_t channel.
 * This is on purpose, to allow it to be connectable in @c stdexec::sync_wait.
 * Therefore, it is not equivalent to @c stdexec::just_stopped.
 *
 * @note Inspired by https://github.com/NVIDIA/stdexec/blob/2184dd714f82df1e00d39961817be97f31c02744/test/test_common/schedulers.hpp#L389-L442.
 */
struct Stopped {
   public:
    using scheduler_concept = stdexec::scheduler_tag;

   private:
    struct Sender;

   public:
    [[nodiscard]]
    constexpr auto schedule() const noexcept -> Sender {
        return {};
    }

    template <stdexec::__one_of<stdexec::set_value_t, stdexec::set_stopped_t> Tag>
    [[nodiscard]]
    constexpr auto query(stdexec::get_completion_scheduler_t<Tag>) const noexcept -> Stopped {
        return {};
    }

    [[nodiscard]]
    friend bool operator==(const Stopped&, const Stopped&) noexcept = default;

   private:
    template <stdexec::receiver Rcvr>
    struct OpState : Kokkos::Execution::Impl::Immovable {
        void start() & noexcept {
            stdexec::set_stopped(std::move(rcvr));
        }

        Rcvr rcvr;
    };

    struct Sender {
        using sender_concept = stdexec::sender_tag;

        using completion_signatures = stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_stopped_t()>;

        template <stdexec::receiver_of<completion_signatures> Rcvr>
            constexpr auto connect(Rcvr rcvr) && KOKKOS_EXECUTION_RETURNS_BRACED(OpState<Rcvr>, {}, std::move(rcvr))

        [[nodiscard]]
        constexpr auto get_env() const noexcept -> Attributes<Stopped, stdexec::set_value_t, stdexec::set_stopped_t> {
            return {};
        }
    };
};

} // namespace Tests::Utils::Schedulers

#endif // KOKKOS_EXECUTION_TESTS_UTILS_SCHEDULERS_STOPPED_HPP
