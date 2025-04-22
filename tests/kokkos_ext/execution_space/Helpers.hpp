#ifndef GRAPH_DISPATCHING_TESTS_KOKKOS_EXT_EXECUTION_SPACE_HELPERS_HPP
#define GRAPH_DISPATCHING_TESTS_KOKKOS_EXT_EXECUTION_SPACE_HELPERS_HPP

#include "gtest/gtest.h"

#include "kokkos-utils/concepts/ExecutionSpace.hpp"

#include "kokkos_ext/impl/ExecutionSpaceContext.hpp"

#include "tests/Functors.hpp"

namespace tests::kokkos_ext
{
namespace impl
{
template <typename Exec>
struct ExecutionSpaceContextTest : public virtual ::testing::Test
{
public:
    using context_t          = Kokkos::Experimental::ExecutionSpaceContext<Exec>;
    using scheduler_t        = decltype(std::declval<const context_t>().get_scheduler());
    using schedule_sender_t  = decltype(::stdexec::schedule(std::declval<scheduler_t>()));
    using scheduler_domain_t = std::invoke_result_t<::stdexec::get_domain_t, scheduler_t>;

    using view_s_t = Kokkos::View<int, Kokkos::SharedSpace>;

public:
    ExecutionSpaceContextTest()
        : exec(Kokkos::Experimental::partition_space(Exec{}, 1)[0])
    {}

protected:
    Exec exec {};
};

} // namespace impl

//! Add a @c then using @ref tests::ThenFunctor that may throw.
#define ADD_THEN ::stdexec::then(ThenFunctor<std::remove_cvref_t<decltype(data)>, true>{.data = data})

//! Get the dispatch label from @p Exec and @p label.
template <Kokkos::utils::concepts::ExecutionSpace Exec, typename Label>
constexpr std::string dispatch_label(const Exec&, Label&& label) {
    return std::string(Kokkos::Impl::TypeInfo<Exec>::name()).append(": ").append(std::forward<Label>(label));
}

} // namespace tests::kokkos_ext

#endif // GRAPH_DISPATCHING_TESTS_KOKKOS_EXT_EXECUTION_SPACE_HELPERS_HPP
