#include "gtest/gtest.h"

#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-declarations")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
#include "kokkos-execution/stdexec/P3955/lifetime.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "tests/utils/sink_receiver.hpp"

namespace Tests {

//! Inspired by the examples from @cite P3955R2. It calls @c T::exit when the scope ends.
template <typename T> requires ( std::constructible_from<T, size_t*> && requires (T& obj) {
    {obj.exit() } -> std::same_as<void>;
})
struct scope_object_with_exit {
    using type = T;

    size_t* counter;

    exec::enter_scope_sender auto operator()(type* storage) const noexcept {
        return stdexec::just() | stdexec::then([counter = counter, storage = storage]() {
                   auto* ptr = std::construct_at(storage, counter);
                   return stdexec::just() | stdexec::then([ptr]() noexcept { ptr->exit(); });
               });
    }
};

struct DoSomethingOnExit {
    size_t* counter;

    void exit() noexcept {
        std::printf("I just reached the finish line of my lifetime.\n"); // NOLINT(modernize-use-std-print)
        ++*counter;
    }
};

//! @test Check that @c exec::lifetime works as expected.
TEST(lifetime, scope_object_with_exit) {
    size_t counter = 0;

    stdexec::sync_wait(
        exec::lifetime(
            [&counter](DoSomethingOnExit&) {
                EXPECT_EQ(counter, 0);
                return stdexec::just();
            },
            scope_object_with_exit<DoSomethingOnExit>{.counter = std::addressof(counter)}));

    ASSERT_EQ(counter, 1);
}

} // namespace Tests
