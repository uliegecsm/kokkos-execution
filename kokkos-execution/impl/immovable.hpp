#ifndef KOKKOS_EXECUTION_IMPL_IMMOVABLE_HPP
#define KOKKOS_EXECUTION_IMPL_IMMOVABLE_HPP

namespace Kokkos::Execution::Impl {
/**
 * @brief Immovable type.
 *
 * Heavily inspired by
 * https://github.com/NVIDIA/stdexec/blob/47bb920c84be5bdb31e6a1d0f8c47ac6e7d54588/include/stdexec/__detail/__utility.hpp#L60-L64.
 *
 * However, commit
 * https://github.com/NVIDIA/stdexec/commit/aab5da8b7f7ed60053d16798a687d0b1549e2e1d
 * changed how @c STDEXEC_IMMOVABLE is defined for @c GCC, see
 * https://github.com/NVIDIA/stdexec/blame/47bb920c84be5bdb31e6a1d0f8c47ac6e7d54588/include/stdexec/__detail/__config.hpp#L548,
 * that is used in the definition of @c stdexec::__immovable.
 *
 * This implementation avoids relying on such internals by providing a minimal, self-contained definition.
 */
struct Immovable {
    Immovable() = default;
    Immovable(Immovable&&) = delete;
    Immovable& operator=(Immovable&&) = delete;
    Immovable(const Immovable&) = delete;
    Immovable& operator=(const Immovable&) = delete;
    ~Immovable() = default;
};

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_IMMOVABLE_HPP
