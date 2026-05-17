#ifndef KOKKOS_EXECUTION_IMPL_SENDER_INTROSPECTION_HPP
#define KOKKOS_EXECUTION_IMPL_SENDER_INTROSPECTION_HPP

#include "kokkos-execution/stdexec.hpp"

namespace Kokkos::Execution::Impl {

/**
 * @brief Retrieve the completion scheduler for a given completion tag.
 *
 * It is heavily inspired, yet different from
 * https://github.com/NVIDIA/stdexec/blob/45c0f5803c190366a8529833901d1f6340b40d2e/include/stdexec/__detail/__schedulers.hpp#L395-L398.
 *
 * While @c stdexec::__completion_scheduler_of_t is constrained to a sender that "sends", this version does not and returns
 * what a genuine call to @c stdexec::get_completion_scheduler would.
 */
template <typename Tag, stdexec::sender Sndr, typename... Env>
using completion_scheduler_of_t =
    std::invoke_result_t<stdexec::get_completion_scheduler_t<Tag>, stdexec::env_of_t<Sndr>, Env...>;

//! Check that a sender has one child.
template <typename Sndr>
concept has_child = requires {
    typename stdexec::__desc_of_t<Sndr>;
    requires stdexec::__nbr_children_of<Sndr> == 1;
};

//! Check that a sender has more than one child.
template <typename Sndr>
concept has_children = requires {
    typename stdexec::__desc_of_t<Sndr>;
    requires stdexec::__nbr_children_of<Sndr> > 1;
};

struct no_env_provided_t { };

template <typename Tag, typename Sndr, typename Env>
struct completion_domain_of {
    using type = stdexec::__completion_domain_of_t<Tag, Sndr, Env>;
};

template <typename Tag, typename Sndr>
struct completion_domain_of<Tag, Sndr, no_env_provided_t> {
    using type = stdexec::__completion_domain_of_t<Tag, Sndr>;
};

template <typename Tag, typename Sndr, typename Env>
using completion_domain_of_t = typename completion_domain_of<Tag, Sndr, Env>::type;

template <typename Tag, stdexec::sender Sndr, typename Domain, typename Env, typename... Conditions>
struct RemainsOn;

template <typename Tag, stdexec::sender Sndr, typename Domain, typename Env, typename... Conditions>
requires(!has_child<Sndr> && !has_children<Sndr>)
struct RemainsOn<Tag, Sndr, Domain, Env, Conditions...> {
    static constexpr bool value = std::same_as<completion_domain_of_t<Tag, Sndr, Env>, Domain>;
};

template <typename Tag, stdexec::sender Sndr, typename Domain, typename Env, typename... Conditions>
requires(has_child<Sndr>)
struct RemainsOn<Tag, Sndr, Domain, Env, Conditions...> {
    static constexpr bool value = std::same_as<completion_domain_of_t<Tag, Sndr, Env>, Domain>
                               && RemainsOn<Tag, stdexec::__child_of<Sndr>, Domain, Env, Conditions...>::value;
};

template <typename Tag, stdexec::sender Sndr, typename Domain, typename Env, typename... Conditions>
requires(has_children<Sndr>)
struct RemainsOn<Tag, Sndr, Domain, Env, Conditions...> {
    template <typename Child>
    using query_t = RemainsOn<Tag, Child, Domain, Env, Conditions...>;

    static constexpr bool value =
        stdexec::__mapply<stdexec::__mall_of<stdexec::__q<query_t>>, stdexec::__children_of<Sndr>>::value;
};

//! Analyse the sender recursively to check if it remains on the same domain.
template <typename Tag, typename Sndr, typename Domain, typename Env = no_env_provided_t, typename... Conditions>
concept remains_on = RemainsOn<Tag, Sndr, Domain, Env, Conditions...>::value;

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_SENDER_INTROSPECTION_HPP
