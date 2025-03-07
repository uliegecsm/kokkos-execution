#ifndef GRAPH_DISPATCHING_TESTS_KOKKOS_EXT_EXECUTION_SPACE_HELPERS_HPP
#define GRAPH_DISPATCHING_TESTS_KOKKOS_EXT_EXECUTION_SPACE_HELPERS_HPP

#include "gtest/gtest.h"

#include "plog/Appenders/ConsoleAppender.h"
#include "plog/Formatters/MessageOnlyFormatter.h"
#include "plog/Init.h"

#include "kokkos_ext/impl/ExecutionSpaceContext.hpp"

namespace tests::kokkos_ext
{
namespace impl
{
template <typename Exec>
struct ExecutionSpaceContextTest : public ::testing::Test
{
public:
    using context_t          = Kokkos::Experimental::ExecutionSpaceContext<Exec>;
    using scheduler_t        = decltype(std::declval<const context_t>().get_scheduler());
    using schedule_sender_t  = decltype(::stdexec::schedule(std::declval<scheduler_t>()));
    using scheduler_domain_t = std::invoke_result_t<::stdexec::get_domain_t, scheduler_t>;

public:
    static void SetUpTestSuite()
    {
        plog::init(plog::verbose, &console_appender);
        exec = Kokkos::Experimental::partition_space(Exec{}, 1)[0];
    }

    static void TearDownTestSuite() { exec.reset(); }

protected:
    static inline std::optional<Exec> exec = std::nullopt;
    static inline plog::ConsoleAppender<plog::MessageOnlyFormatter> console_appender;
};

} // namespace impl

} // namespace tests::kokkos_ext

#endif // GRAPH_DISPATCHING_TESTS_KOKKOS_EXT_EXECUTION_SPACE_HELPERS_HPP
