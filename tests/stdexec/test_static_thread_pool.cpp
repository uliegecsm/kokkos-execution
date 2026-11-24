#include "gtest/gtest.h"

#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/static_thread_pool.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-execution/stdexec.hpp"

/**
 * @addtogroup unittests
 *
 * Check the behavior of @c experimental::execution::static_thread_pool
 * --------------------------------------------------------------------
 *
 * This group of tests check the behavior of @c experimental::execution::static_thread_pool.
 *
 * The tests can be found in @ref tests/stdexec/test_static_thread_pool.cpp.
 */

namespace Tests {

using scheduler_t = experimental::execution::static_thread_pool::scheduler;
using schd_sndr_t = stdexec::schedule_result_t<scheduler_t>;

/**
 * @test Check that the @c experimental::execution::static_thread_pool::domain transforms @c stdexec::bulk.
 *
 * See also:
 *  - https://github.com/NVIDIA/stdexec/blob/4781797199f20ef742b39ea00785a1503badd218/include/stdexec/__detail/__transform_sender.hpp#L80-L195
 *  - https://github.com/NVIDIA/stdexec/blob/5d229d49595ccd306cdd160b53ad1a018d137810/include/stdexec/__detail/__bulk.hpp#L434-L444
 *  - https://github.com/NVIDIA/stdexec/blob/5d229d49595ccd306cdd160b53ad1a018d137810/include/exec/thread_pool_base.hpp#L649-L699
 */
consteval bool test_static_thread_pool_domain_transforms_bulk() {
    using sndr_t = decltype(std::declval<schd_sndr_t>() | stdexec::bulk(stdexec::par, 42, [](const int) { }));

    using completion_domain_t = stdexec::__completion_domain_of_t<stdexec::set_value_t, sndr_t, stdexec::env<>>;
    static_assert(std::same_as<completion_domain_t, experimental::execution::static_thread_pool::domain>);

    //! The static thread pool domain transforms the sender.
    using trnsfrmd_sndr_t = decltype(stdexec::transform_sender(std::declval<sndr_t>(), stdexec::env<>{}));
    static_assert(!std::is_reference_v<trnsfrmd_sndr_t>);
    static_assert(!std::same_as<trnsfrmd_sndr_t, sndr_t>);

    //! However, the static thread pool domain itself does not provide a @c stdexec::transform_sender customization for @c stdexec::bulk.
    static_assert(
        !stdexec::__detail::__has_transform_sender<completion_domain_t, stdexec::set_value_t, sndr_t, stdexec::env<>>);

    //! Instead, the transformation is recursive. The default domain does not return the sender as it is.
    static_assert(!stdexec::__detail::__transform_sender<
                  completion_domain_t,
                  stdexec::set_value_t
    >::template __leaves_alone<sndr_t, stdexec::env<>>);

    static_assert(stdexec::__sender_for<sndr_t, stdexec::bulk_t>);

    static_assert(stdexec::__detail::__has_tag_transform_sender<stdexec::set_value_t, sndr_t, stdexec::env<>>);

    static_assert(stdexec::__detail::__has_transform_sender<
                  stdexec::default_domain,
                  stdexec::set_value_t,
                  sndr_t,
                  stdexec::env<>
    >);

    using trnsfrmd_sndr_default_t = stdexec::__detail::__transform_sender_result_t<
        stdexec::default_domain,
        stdexec::set_value_t,
        sndr_t,
        stdexec::env<>
    >;

    static_assert(stdexec::__sender_for<trnsfrmd_sndr_default_t, stdexec::bulk_chunked_t>);

    static_assert(!std::same_as<trnsfrmd_sndr_default_t, trnsfrmd_sndr_t>);

    //! The static thread pool domain does transform the sender returned by the default domain.
    static_assert(stdexec::__detail::__has_transform_sender<
                  completion_domain_t,
                  stdexec::set_value_t,
                  trnsfrmd_sndr_default_t,
                  stdexec::env<>
    >);

    using trnsfrmd_sndr_thread_pool_after_default_t = stdexec::__detail::__transform_sender_result_t<
        completion_domain_t,
        stdexec::set_value_t,
        trnsfrmd_sndr_default_t,
        stdexec::env<>
    >;

    static_assert(std::same_as<trnsfrmd_sndr_thread_pool_after_default_t, trnsfrmd_sndr_t>);

    return true;
}

static_assert(test_static_thread_pool_domain_transforms_bulk());

//! Because we need at least one discoverable runtime test per unit.
TEST(is, useless) {
}

} // namespace Tests
