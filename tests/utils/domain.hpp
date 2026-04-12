#ifndef KOKKOS_EXECUTION_TESTS_UTILS_DOMAIN_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_DOMAIN_HPP

#include "kokkos-execution/stdexec.hpp"

#include "tests/utils/functors/no_op.hpp"

namespace Tests::Utils {

struct DomainInheritingFromDefault : public stdexec::default_domain { };

//! Check that a domain has the @c stdexec::default_domain as common domain with a few other domains.
template <typename DomainType>
consteval bool check_common_domain_is_default() {
    static_assert(
        std::same_as<stdexec::__common_domain_t<DomainType, stdexec::default_domain>, stdexec::default_domain>);

    static_assert(
        std::same_as<stdexec::__common_domain_t<DomainType, DomainInheritingFromDefault>, stdexec::default_domain>);

    return true;
}

/**
 * @brief Check if the domain's transform of a @c stdexec::then sender is "default-like".
 *
 * Inspired by https://github.com/NVIDIA/stdexec/blob/b06fe4b00d25f226c87d23e7ba1e564c2c878a15/include/stdexec/__detail/__domain.hpp#L98-L102.
 */
template <typename DomainType, stdexec::sender ScheduleSenderType>
consteval bool check_if_default_domain_like_then() {
    using functor_t = Tests::Utils::Functors::NoOp<false, false, false>;
    using sndr_t = decltype(stdexec::then(std::declval<ScheduleSenderType>(), std::declval<functor_t>()));

    static_assert(stdexec::__detail::__has_transform_sender<DomainType, stdexec::set_value_t, sndr_t, stdexec::env<>>);

    return stdexec::__default_domain_like<DomainType, stdexec::set_value_t, sndr_t, stdexec::env<>>;
}

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_DOMAIN_HPP
