#ifndef KOKKOS_EXECUTION_TESTS_UTILS_SCHEDULERS_INLINE_SCHEDULER_WITH_DOMAIN_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_SCHEDULERS_INLINE_SCHEDULER_WITH_DOMAIN_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/impl/immovable.hpp"

namespace Tests::Utils::Schedulers {

template <typename Domain>
struct InlineSchedulerWithDomain {
    using scheduler_concept = stdexec::scheduler_tag;

    template <stdexec::receiver Rcvr>
    struct OpState : public Kokkos::Execution::Impl::Immovable {
        using operation_state_concept = stdexec::operation_state_tag;

        void start() & noexcept {
            stdexec::set_value(std::move(rcvr));
        }

        Rcvr rcvr;
    };

    struct Sender {
        using sender_concept = stdexec::sender_tag;

        using completion_signatures = stdexec::completion_signatures<stdexec::set_value_t()>;

        struct Attributes {
            [[nodiscard]]
            constexpr auto query(stdexec::get_completion_scheduler_t<stdexec::set_value_t>) const noexcept
                -> InlineSchedulerWithDomain {
                return {};
            }

            [[nodiscard]]
            constexpr auto query(stdexec::get_completion_domain_t<stdexec::set_value_t>) const noexcept -> Domain {
                return {};
            }
        };

        template <stdexec::receiver_of<completion_signatures> Rcvr>
        auto connect(Rcvr rcvr) && noexcept(std::is_nothrow_move_constructible_v<Rcvr>) -> OpState<Rcvr> {
            return {{}, std::move(rcvr)};
        }

        [[nodiscard]]
        constexpr auto get_env() const noexcept -> Attributes {
            return {};
        }
    };

    [[nodiscard]]
    constexpr auto schedule() const noexcept -> Sender {
        return {};
    }

    [[nodiscard]]
    constexpr auto query(stdexec::get_completion_domain_t<stdexec::set_value_t>) const noexcept -> Domain {
        return {};
    }

    [[nodiscard]]
    constexpr auto
        query(stdexec::get_completion_scheduler_t<stdexec::set_value_t>) const noexcept -> InlineSchedulerWithDomain {
        return {};
    }

    friend auto operator<=>(const InlineSchedulerWithDomain&, const InlineSchedulerWithDomain&) noexcept = default;
};

} // namespace Tests::Utils::Schedulers

#endif // KOKKOS_EXECUTION_TESTS_UTILS_SCHEDULERS_INLINE_SCHEDULER_WITH_DOMAIN_HPP
