#ifndef KOKKOS_EXECUTION_IMPL_CONTINUES_ON_HPP
#define KOKKOS_EXECUTION_IMPL_CONTINUES_ON_HPP

#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/fork_join.hpp"
PRAGMA_DIAGNOSTIC_POP

namespace Kokkos::Execution::ExecutionSpaceImpl {

//! Determine if at least one of the children of the sender completes on the given domain.
template <typename ChannelTag, typename DomainType, typename Sndr, typename Env>
struct has_at_least_one_child_completing_on {
    template <typename T>
    using completes_on_domain =
        std::bool_constant<std::same_as<DomainType, stdexec::__detail::__completing_domain_t<ChannelTag, T, Env>>>;

   public:
    static constexpr bool value =
        stdexec::__mapply<stdexec::__many_of<stdexec::__q<completes_on_domain>>, stdexec::__children_of<Sndr>>::value;
};

//! Check that a sender has a single child with a given tag.
template <typename Sndr, typename Tag>
concept has_single_child_with_tag = requires {
    requires stdexec::__nbr_children_of<Sndr> == 1;
    typename stdexec::tag_of_t<stdexec::__child_of<Sndr>>;
    requires std::same_as<stdexec::tag_of_t<stdexec::__child_of<Sndr>>, Tag>;
};

//! Determine if the sender has a single @c stdexec::when_all child, that has at least one branch completing on the given domain.
template <typename ChannelTag, typename DomainType, typename Sndr, typename Env>
struct has_when_all_child_with_at_least_one_child_completing_on {
    static constexpr bool value = false;
};

template <typename ChannelTag, typename DomainType, has_single_child_with_tag<stdexec::when_all_t> Sndr, typename Env>
struct has_when_all_child_with_at_least_one_child_completing_on<ChannelTag, DomainType, Sndr, Env> {
   private:
    using child_t = stdexec::__child_of<Sndr>;

   public:
    static constexpr bool value = has_at_least_one_child_completing_on<ChannelTag, DomainType, child_t, Env>::value;
};

template <typename ChannelTag, typename DomainType, typename Sndr, typename Env>
inline constexpr bool has_when_all_child_with_at_least_one_child_completing_on_v =
    has_when_all_child_with_at_least_one_child_completing_on<ChannelTag, Domain, Sndr, Env>::value;

//! Determine if the sender has a single @c exec::fork_join child, that has at least one branch completing on the given domain.
template <typename ChannelTag, typename DomainType, typename Sndr, typename Env>
struct has_fork_join_child_with_at_least_one_child_completing_on {
    static constexpr bool value = false;
};

template <typename ChannelTag, typename DomainType, has_single_child_with_tag<exec::fork_join_t> Sndr, typename Env>
struct has_fork_join_child_with_at_least_one_child_completing_on<ChannelTag, DomainType, Sndr, Env> {
    using child_t = stdexec::__child_of<Sndr>;

    static consteval bool check() noexcept {
        using _closures_t = stdexec::__data_of<child_t>;
        using _child_sndr_t = stdexec::__child_of<child_t>;
        using _domain_t = stdexec::__completion_domain_of_t<ChannelTag, _child_sndr_t, Env>;
        using _child_t = stdexec::__copy_cvref_t<child_t, _child_sndr_t>;
        using _child_completions_t = stdexec::__completion_signatures_of_t<_child_t, stdexec::__fwd_env_t<Env>>;
        using _sndr_t = exec::fork_join_impl_t::_when_all_sndr_t<_child_completions_t, _closures_t, _domain_t>;
        return has_at_least_one_child_completing_on<ChannelTag, DomainType, _sndr_t, Env>::value;
    }

   public:
    static constexpr bool value = check();
};

template <typename ChannelTag, typename DomainType, typename Sndr, typename Env>
inline constexpr bool has_fork_join_child_with_at_least_one_child_completing_on_v =
    has_fork_join_child_with_at_least_one_child_completing_on<ChannelTag, DomainType, Sndr, Env>::value;

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_IMPL_CONTINUES_ON_HPP
