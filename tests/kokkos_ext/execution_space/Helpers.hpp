#ifndef GRAPH_DISPATCHING_TESTS_KOKKOS_EXT_EXECUTION_SPACE_HELPERS_HPP
#define GRAPH_DISPATCHING_TESTS_KOKKOS_EXT_EXECUTION_SPACE_HELPERS_HPP

#include "gtest/gtest.h"

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

#define MATCHER_FOR_NAME(__type__, __name__)                                               \
    ::testing::Field(                                                                      \
        &Kokkos::utils::callbacks::__type__##Event::name,                                  \
        ::testing::StrEq(                                                                  \
            std::format("{}: " #__name__, Kokkos::Impl::TypeInfo<execution_space>::name()) \
        )                                                                                  \
    )

#define MATCHER_FOR_DEV_ID(__type__)                        \
    ::testing::Field(                                       \
        &Kokkos::utils::callbacks::__type__##Event::dev_id, \
        ::testing::Eq(                                      \
            Kokkos::Tools::Experimental::device_id(exec)    \
        )                                                   \
    )

#define MATCHER_FOR_BEGIN_FENCE ABeginFenceEvent      (MATCHER_FOR_NAME(BeginFence,       sync_wait), MATCHER_FOR_DEV_ID(BeginFence))
#define MATCHER_FOR_BEGIN_PFOR  ABeginParallelForEvent(MATCHER_FOR_NAME(BeginParallelFor, then),      MATCHER_FOR_DEV_ID(BeginParallelFor))

} // namespace tests::kokkos_ext

#endif // GRAPH_DISPATCHING_TESTS_KOKKOS_EXT_EXECUTION_SPACE_HELPERS_HPP
