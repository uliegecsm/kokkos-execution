#ifndef GRAPH_DISPATCHING_TESTS_KOKKOS_EXT_EXECUTION_SPACE_HELPERS_HPP
#define GRAPH_DISPATCHING_TESTS_KOKKOS_EXT_EXECUTION_SPACE_HELPERS_HPP

#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"

#include "kokkos-utils/concepts/ExecutionSpace.hpp"
#include "kokkos-utils/tests/scoped/ExecutionSpace.hpp"

#include "kokkos_ext/impl/ExecutionSpaceContext.hpp"

#include "tests/Functors.hpp"

namespace tests::kokkos_ext
{
namespace impl
{
template <typename Exec>
struct ExecutionSpaceContextTest : public virtual ::testing::Test,
                                   public Kokkos::utils::tests::scoped::ExecutionSpace<Exec>
{
public:
    using context_t          = Kokkos::Experimental::ExecutionSpaceContext<Exec>;
    using scheduler_t        = decltype(std::declval<const context_t>().get_scheduler());
    using schedule_sender_t  = decltype(::stdexec::schedule(std::declval<scheduler_t>()));
    using scheduler_domain_t = std::invoke_result_t<::stdexec::get_domain_t, scheduler_t>;

    using view_s_t = Kokkos::View<int, Kokkos::SharedSpace>;
};

} // namespace impl

//! Add a @c then using @ref tests::ThenFunctor that may throw.
#define ADD_THEN ::stdexec::then(ThenFunctor<std::remove_cvref_t<decltype(data)>, true>{.data = data})

//! Get the dispatch label from @p Exec and @p label.
template <Kokkos::utils::concepts::ExecutionSpace Exec, typename Label>
constexpr std::string dispatch_label(const Exec&, Label&& label) {
    return std::string(Kokkos::Impl::TypeInfo<Exec>::name()).append(": ").append(std::forward<Label>(label));
}

template<typename T1, typename T2>
constexpr bool are_same_instances(const T1& lhs, const T2& rhs) {
    if constexpr (std::same_as<T1, T2>) {
        return lhs == rhs;
    } else {
        return false;
    }
}

} // namespace tests::kokkos_ext

#endif // GRAPH_DISPATCHING_TESTS_KOKKOS_EXT_EXECUTION_SPACE_HELPERS_HPP
