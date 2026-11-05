#ifndef KOKKOS_EXECUTION_TESTS_UTILS_SCHEDULERS_ATTRIBUTES_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_SCHEDULERS_ATTRIBUTES_HPP

#include "kokkos-execution/stdexec.hpp"

namespace Tests::Utils::Schedulers {

//! Inspired by https://github.com/NVIDIA/stdexec/blob/2184dd714f82df1e00d39961817be97f31c02744/test/test_common/schedulers.hpp#L46.
template <typename Schd, stdexec::__completion_tag... Tags>
struct Attributes {
    static_assert(stdexec::scheduler<Schd>);

    template <stdexec::__one_of<Tags...> Tag, typename... Env>
    [[nodiscard]]
    constexpr auto query(stdexec::get_completion_scheduler_t<Tag>, const Env&...) const noexcept {
        return m_schd;
    }

    Schd m_schd;
};

} // namespace Tests::Utils::Schedulers

#endif // KOKKOS_EXECUTION_TESTS_UTILS_SCHEDULERS_ATTRIBUTES_HPP
