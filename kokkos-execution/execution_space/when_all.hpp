#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_WHEN_ALL_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_WHEN_ALL_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/execution_space/execution_space_fwd.hpp"

#include "kokkos-execution/execution_space/env.hpp"
#include "kokkos-execution/impl/dependency.hpp"
#include "kokkos-execution/impl/env.hpp"
#include "kokkos-execution/impl/get_exec.hpp"
#include "kokkos-execution/impl/receiver.hpp"
#include "kokkos-execution/impl/submitted.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

template <Kokkos::ExecutionSpace Exec, size_t Idx, typename ParentOp, typename Env = stdexec::env_of_t<ParentOp>>
struct WhenAllChildReceiver : public Impl::Receiver<ParentOp, Env> {
    using exec_env_policy_t = typename ParentOp::exec_env_policy_t;

    void submitted(Impl::OptionalConstEventRef<Exec> dep) && noexcept {
        this->parent_op->template submit<Idx>(dep);
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> extend_env_with_exec_t<exec_env_policy_t, Env> {
        return extend_env_with_exec<exec_env_policy_t>(stdexec::get_env(*this->parent_op));
    }
};

enum class WhenAllState : std::uint8_t {
    started,
    error,
    stopped
};

template <typename ChildOpStates, typename ChildExecs, typename Rcvr>
struct WhenAllSignalsSubmitted;

template <typename... OpStates, typename... Execs, typename Rcvr>
struct WhenAllSignalsSubmitted<stdexec::__tuple<OpStates...>, stdexec::__tuple<Execs...>, Rcvr> {
    static constexpr bool value = (Impl::signals_submitted<OpStates> || ...)
                               && Impl::supports_submitted_depend_on<Rcvr, Execs...>;
};

template <typename ChildOpStates, typename ChildExecs, typename Rcvr>
concept when_all_signals_submitted = WhenAllSignalsSubmitted<ChildOpStates, ChildExecs, Rcvr>::value;

template <stdexec::receiver Rcvr>
struct WhenAllOpStateBase {
    Rcvr rcvr;

    KOKKOS_EXECUTION_GET_ENV(Rcvr, this->rcvr)
};

template <stdexec::receiver Rcvr, stdexec::sender... Sndrs>
struct WhenAllOpState
    : public Impl::Immovable
    , public WhenAllOpStateBase<Rcvr> {
    using operation_state_concept = Impl::SubmittedOperationStateTag;

    using base_t = WhenAllOpStateBase<Rcvr>;

    static constexpr bool sends_stopped = (stdexec::sends_stopped<Sndrs, stdexec::env_of_t<Rcvr>> || ...);

    using exec_env_policy_t = extend_env_with_exec_policy_t<stdexec::env_of_t<Rcvr>>;

    using child_execs_t = stdexec::__tuple<Impl::exec_of_t<Sndrs, stdexec::env_of_t<Rcvr>>...>;

    template <typename ChildExecs>
    struct ChildEvents;

    template <typename... Execs>
    struct ChildEvents<stdexec::__tuple<Execs...>> {
        using type = stdexec::__tuple<Impl::OptionalConstEventRef<Execs>...>;
    };

    using child_events_t = typename ChildEvents<child_execs_t>::type;

    template <typename IndexSequence>
    struct Connect;

    template <size_t... Idxs>
    struct Connect<std::index_sequence<Idxs...>> {
        constexpr auto operator()(WhenAllOpState* opstate, Sndrs&&... sndrs) const noexcept(
            (stdexec::__nothrow_connectable<
                 Sndrs,
                 WhenAllChildReceiver<
                     Impl::exec_of_t<Sndrs, stdexec::env_of_t<Rcvr>>,
                     Idxs,
                     WhenAllOpState,
                     stdexec::env_of_t<Rcvr>
                 >
             >
             && ...))
            -> stdexec::__tuple<stdexec::connect_result_t<
                Sndrs,
                WhenAllChildReceiver<
                    Impl::exec_of_t<Sndrs, stdexec::env_of_t<Rcvr>>,
                    Idxs,
                    WhenAllOpState,
                    stdexec::env_of_t<Rcvr>
                >
            >...> {
            return {stdexec::connect(
                std::forward<Sndrs>(sndrs),
                WhenAllChildReceiver<
                    Impl::exec_of_t<Sndrs, stdexec::env_of_t<Rcvr>>,
                    Idxs,
                    WhenAllOpState,
                    stdexec::env_of_t<Rcvr>
                >{opstate})...};
        }
    };

    using child_opstates_t =
        std::invoke_result_t<Connect<std::index_sequence_for<Sndrs...>>, WhenAllOpState*, Sndrs&&...>;

    using completion_signal_policy_t = std::conditional_t<
        when_all_signals_submitted<child_opstates_t, child_execs_t, Rcvr>,
        Impl::SubmittedPolicy::DependOnEvent,
        Impl::SyncPolicy::InlineFenceExec
    >;

    explicit WhenAllOpState(
        Sndrs&&... sndrs, // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
        Rcvr rcvr)
        noexcept(
            std::is_nothrow_constructible_v<base_t, Rcvr&&>
            && std::is_nothrow_invocable_v<Connect<std::index_sequence_for<Sndrs...>>, WhenAllOpState*, Sndrs&&...>)
        : base_t(std::move(rcvr))
        , count(sizeof...(Sndrs))
        , child_opstates(Connect<std::index_sequence_for<Sndrs...>>{}(this, std::forward<Sndrs>(sndrs)...)) {
    }

    void complete(stdexec::set_value_t) noexcept {
        this->arrive();
    }

    template <typename Error>
    void complete(stdexec::set_error_t, Error&& error) noexcept {
        if (this->state.exchange(WhenAllState::error, std::memory_order_relaxed) != WhenAllState::error) {
            if constexpr (std::same_as<std::remove_cvref_t<Error>, std::exception_ptr>) {
                this->m_error = error;
            } else if constexpr (std::is_nothrow_constructible_v<std::remove_cvref_t<Error>, Error>) {
                this->m_error = std::make_exception_ptr(std::forward<Error>(error));
            } else {
                try {
                    this->m_error = std::make_exception_ptr(std::forward<Error>(error));
                } catch (...) {
                    this->m_error = std::current_exception();
                }
            }
        }
        this->arrive();
    }

    void complete(stdexec::set_stopped_t) noexcept {
        WhenAllState expected = WhenAllState::started;
        this->state.compare_exchange_strong(expected, WhenAllState::stopped, std::memory_order_relaxed);
        this->arrive();
    }

    template <size_t Idx>
    void submit(stdexec::__tuple_element_t<Idx, child_events_t> dep) noexcept {
        stdexec::__get<Idx>(this->events) = dep;
        this->arrive();
    }

    void arrive() noexcept {
        if (1 == this->count.fetch_sub(1, std::memory_order_acq_rel)) {
            this->submit();
        }
    }

    void submit() noexcept {
        switch (this->state.load(std::memory_order_relaxed)) {
        case WhenAllState::started:
            if constexpr (std::same_as<completion_signal_policy_t, Impl::SubmittedPolicy::DependOnEvent>) {
                stdexec::__apply(
                    [&](const auto&... deps) noexcept { std::move(this->rcvr).submitted(deps...); }, this->events);
            } else {
                static_assert(std::same_as<completion_signal_policy_t, Impl::SyncPolicy::InlineFenceExec>);
                try {
                    stdexec::__apply([](const auto&... deps) { Impl::wait_on(deps...); }, this->events);
                } catch (...) {
                    stdexec::set_error(std::move(this->rcvr), std::current_exception());
                    return;
                }
                stdexec::set_value(std::move(this->rcvr));
            }
            break;
        case WhenAllState::error:
            stdexec::set_error(std::move(this->rcvr), *this->m_error);
            break;
        case WhenAllState::stopped:
            if constexpr (sends_stopped) {
                stdexec::set_stopped(std::move(this->rcvr));
            } else {
                STDEXEC_UNREACHABLE();
            }
            break;
        default:
            STDEXEC_UNREACHABLE();
        }
    }

    void start() & noexcept {
        stdexec::__apply([](auto&... opstates) noexcept -> void { (stdexec::start(opstates), ...); }, child_opstates);
    }

    std::atomic<size_t> count;
    std::atomic<WhenAllState> state{WhenAllState::started};
    child_events_t events{};
    std::optional<std::exception_ptr> m_error{};
    child_opstates_t child_opstates;
};

template <stdexec::sender... Sndrs>
struct WhenAllSender {
    using sender_concept = stdexec::sender_tag;

    using sndrs_t = stdexec::__tuple<Sndrs...>;

    struct attrs {
        template <typename... Env>
        [[nodiscard]]
        constexpr auto
            query(stdexec::get_completion_domain_t<stdexec::set_value_t>, const Env&...) const noexcept -> Domain {
            return {};
        }
    };

    template <stdexec::__decays_to<WhenAllSender> Self, typename... Env>
    static consteval auto get_completion_signatures() {
        if constexpr (sizeof...(Env) == 0 && (stdexec::dependent_sender<Sndrs> || ...)) {
            return stdexec::__throw_dependent_sender_error<Self>();
        } else {
            return stdexec::__concat_completion_signatures(
                stdexec::__if_c<
                    (stdexec::sender_of<Sndrs, stdexec::set_value_t(), Env...> && ...),
                    stdexec::completion_signatures<stdexec::set_value_t()>,
                    stdexec::completion_signatures<>
                >{},
                stdexec::completion_signatures<stdexec::set_error_t(std::exception_ptr)>{},
                stdexec::__if_c<
                    (stdexec::sends_stopped<Sndrs, Env...> || ...),
                    stdexec::completion_signatures<stdexec::set_stopped_t()>,
                    stdexec::completion_signatures<>
                >{});
        }
    }

    template <stdexec::receiver Rcvr>
    constexpr auto connect(Rcvr rcvr) && noexcept(
        std::is_nothrow_constructible_v<WhenAllOpState<Rcvr, Sndrs...>, Sndrs&&..., Rcvr&&>)
        -> WhenAllOpState<Rcvr, Sndrs...> {
        return stdexec::__apply(
            [&](auto&&... children) {
                return WhenAllOpState<Rcvr, Sndrs...>(std::forward<Sndrs>(children)..., std::move(rcvr));
            },
            std::move(sndrs));
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> attrs {
        return {};
    }

    sndrs_t sndrs;
};

template <>
struct TransformSenderFor<stdexec::when_all_t> {
    template <typename Env, typename... Sndrs>
    auto operator()(const Env&, stdexec::when_all_t, stdexec::__, Sndrs&&... sndrs) const {
        return WhenAllSender<Sndrs...>{.sndrs = {std::forward<Sndrs>(sndrs)...}};
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_WHEN_ALL_HPP
