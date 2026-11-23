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

template <typename...>
struct TypesForClosure; // check if tag_of works?

template <typename Tag, typename... Args>
struct TypesForClosure<stdexec::__closure<Tag, Args...>> {
    using tag = Tag;
    using data = stdexec::__tuple<Args...>;
};

template <typename T>
using tag_for_closure_t = typename TypesForClosure<T>::tag;

template <typename T>
using data_for_closure_t = typename TypesForClosure<T>::data;

template <>
struct TransformSenderFor<Kokkos::Execution::path_t> {

    template <typename Env, typename Data, typename Sndr>
    struct Transformer;

    template <typename Env, typename... Clsrs, typename Sndr>
    struct Transformer<Env, stdexec::__tuple<Clsrs...>, Sndr> {
        using execution_space = Impl::exec_of_t<Sndr, Env>;

        using Data = stdexec::__tuple<Clsrs...>;

        using closures_t = stdexec::__tuple<
            typename TransformSenderFor<tag_for_closure_t<std::remove_cvref_t<Clsrs>>>::template closure_for_t<
                tag_for_closure_t<std::remove_cvref_t<Clsrs>>,
                execution_space,
                std::remove_cvref_t<Clsrs>
            >...
        >;

        using sndr_t = PathSender<Sndr, closures_t>;

        auto operator()(const Env& env, stdexec::__tuple<Clsrs...>& data, std::remove_reference_t<Sndr>& sndr) const
            STDEXEC_AUTO_RETURN(
                sndr_t{
                    .clsrs = stdexec::__apply(
                        [&sndr, &env]<typename... T>(T&&... clsr) -> closures_t {
                            // todo don't get state each time
                            auto* const state =
                                stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env)
                                    .state;
                            return stdexec::__mktuple(
                                TransformSenderFor<tag_for_closure_t<std::remove_cvref_t<T>>>::template make_closure<
                                    tag_for_closure_t<std::remove_cvref_t<T>>,
                                    execution_space
                                >(state, std::forward<T>(clsr))...);
                        },
                        stdexec::__forward_like<Data>(data)),
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
