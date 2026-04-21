#ifndef KOKKOS_EXECUTION_TESTS_UTILS_CONTEXT_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_CONTEXT_HPP

#include "gtest/gtest.h"

#include "kokkos-execution/stdexec.hpp"

#include "Kokkos_Core.hpp"

#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
#    include "plog/Formatters/TxtFormatter.h"
#    include "plog/Initializers/ConsoleInitializer.h"
#    include "plog/Log.h"
#endif

#include "kokkos-utils/tests/scoped/ExecutionSpace.hpp"

#include "tests/utils/kokkos.hpp"

namespace Tests::Utils {

template <template <typename> typename ContextType, Kokkos::ExecutionSpace Exec>
struct ContextTest
    : public virtual testing::Test
    , public Kokkos::utils::tests::scoped::ExecutionSpace<Exec> {
   public:
    using context_t = ContextType<Exec>;
    using scheduler_t = decltype(std::declval<const context_t>().get_scheduler());
    using schedule_sender_t = decltype(stdexec::schedule(std::declval<scheduler_t>()));

    using value_t = int;
    using view_s_t = Kokkos::View<value_t, Kokkos::SharedSpace>;

    static constexpr bool on_device = Tests::Utils::on_device<TEST_EXECUTION_SPACE>();

   public:
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
    static void SetUpTestSuite() {
        plog::init<plog::TxtFormatter>(plog::debug, plog::streamStdOut);
    }
#endif
};

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_CONTEXT_HPP
