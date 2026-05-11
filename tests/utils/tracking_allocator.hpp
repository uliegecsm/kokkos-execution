#ifndef KOKKOS_EXECUTION_TESTS_UTILS_TRACKING_ALLOCATOR_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_TRACKING_ALLOCATOR_HPP

#include <atomic>
#include <memory>
#include <utility>

namespace Tests::Utils {

/**
 * @brief A minimal tracking allocator.
 *
 * It atomically counts allocations.
 */
template <typename T>
struct TrackingAllocator {
    using value_type = T;

    std::atomic<size_t>* count;

    T* allocate(std::size_t n) {
        ++(*count);
        return std::allocator<T>{}.allocate(n);
    }

    void deallocate(T* ptr, std::size_t size) {
        std::allocator<T>{}.deallocate(ptr, size);
    }

    friend constexpr auto operator<=>(const TrackingAllocator&, const TrackingAllocator&) noexcept = default;
};

/**
 * Allocate one @c T through @p alloc, copy construct in-place from @p value, then deallocate.
 *
 * @return The value that was copy-constructed.
 */
template <typename Allocator, typename T>
auto round_trip_allocate(Allocator& allocator, T&& value) {
    using traits = std::allocator_traits<Allocator>;

    static_assert(std::same_as<typename traits::value_type, std::remove_cvref_t<T>>);

    auto* ptr = traits::allocate(allocator, 1);

    traits::construct(allocator, ptr, std::forward<T>(value));
    std::remove_cvref_t<T> result = *ptr; // NOLINT(misc-const-correctness)

    traits::destroy(allocator, ptr);
    traits::deallocate(allocator, ptr, 1);

    return result;
}

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_TRACKING_ALLOCATOR_HPP
