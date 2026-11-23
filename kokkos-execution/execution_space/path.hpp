#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_PATH_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_PATH_HPP

#include "kokkos-execution/path.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

template <typename Sndr, typename...>
struct PathSender;

template <typename Sndr, typename... Clsrs>
struct PathSender<Sndr, stdexec::__tuple<Clsrs...>> {
    using sender_concept = stdexec::sender_tag;

    stdexec::__tuple<Clsrs...> clsrs;
    Sndr sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)

    // TODO
    // KOKKOS_EXECUTION_COMPL_SIGS_ADD(PathSender, Sndr, stdexec::set_error_t(std::exception_ptr))

    // template <stdexec::receiver Rcvr>
    // constexpr auto connect(Rcvr rcvr) && { /*noexcept(noexcept(make_op_state_t<Sndr, Rcvr, closure_t>{}(
    //     std::declval<Sndr>(),
    //     std::declval<Rcvr>(),
    //     std::declval<closure_t>()))) -> op_state_t<Sndr, Rcvr, closure_t>*/ {
    //     return make_op_state_t<Sndr, Rcvr, closure_t>{}(std::forward<Sndr>(sndr), std::move(rcvr), std::move(clsr));
    // }

    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(Sndr, sndr)
};

template <stdexec::sender Sndr, typename... Env>
using closure_of_t = typename stdexec::transform_sender_result_t<Sndr, Env...>::closure_t;

template <>
struct TransformSenderFor<Kokkos::Execution::path_t> {

    template <typename Env, typename Data, typename Sndr>
    requires stdexec::__is_instance_of<Data, stdexec::__tuple>
    struct Transformer {
        using execution_space = Impl::exec_of_t<Sndr, Env>;

        template <typename Clsr>
        using fake_sndr_t = std::invoke_result_t<Clsr, stdexec::schedule_result_t<Scheduler<execution_space>>>;

        using fake_sndrs_t = stdexec::__mapply<
            stdexec::__mtransform<stdexec::__mbind_front_q<fake_sndr_t>, stdexec::__q<stdexec::__tuple>>,
            Data
        >;

        static_assert(
            stdexec::__mapply<stdexec::__mall_of<stdexec::__q<Impl::is_dispatching_sender_t>>, fake_sndrs_t>::value);

        using closures_t = stdexec::__mapply<
            stdexec::__mtransform<stdexec::__mbind_back_q<closure_of_t, Env>, stdexec::__q<stdexec::__tuple>>,
            fake_sndrs_t
        >;

        using sndr_t = PathSender<Sndr, closures_t>;

        auto operator()(const Env& env, std::remove_reference_t<Data>& data, std::remove_reference_t<Sndr>& sndr) const
            STDEXEC_AUTO_RETURN(
                sndr_t{
                    .clsrs = stdexec::__apply(
                        [&sndr, &env]<typename... T>(T&&... clsr) -> closures_t {
                            // todo don't get state each time
                            // auto* const state =
                            // stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env)
                            // .state;
                            auto schd_sndr = stdexec::schedule(
                                stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env));
                            return stdexec::__mktuple(
                                stdexec::transform_sender(std::forward<T>(clsr)(std::move(schd_sndr)), env).clsr...);
                        },
                        std::forward<Data>(data)),
                    .sndr = std::forward<Sndr>(sndr)})
    };

    template <typename Env, typename Data, typename Sndr>
    requires(stdexec::__sends<stdexec::set_value_t, Sndr, Env> && execution_space_completing_sender<Sndr, Env>)
    auto operator()(
        const Env& env,
        Kokkos::Execution::path_t,
        Data&& data, // NOLINT(cppcoreguidelines-missing-std-forward)
        Sndr&& sndr) const STDEXEC_AUTO_RETURN(Transformer<Env, Data, Sndr>{}(env, data, sndr))

    template <typename Env, typename Data, typename Sndr>
    requires stdexec::__sends<stdexec::set_value_t, Sndr, Env>
    auto operator()(const Env&, Kokkos::Execution::path_t, Data&&, Sndr&&) const noexcept {
        return no_execution_space_scheduler_in_env<path_t, Sndr, Env>();
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_PATH_HPP
