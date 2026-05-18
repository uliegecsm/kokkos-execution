#ifndef KOKKOS_EXECUTION_IMPL_OPTIONAL_REF_HPP
#define KOKKOS_EXECUTION_IMPL_OPTIONAL_REF_HPP

#include "Kokkos_Assert.hpp"

namespace Kokkos::Execution::Impl {

/**
 * @brief A non-owning, nullable reference to a @p T.
 *
 * @warning The caller is responsible for ensuring the referenced object outlives the @ref m_ptr handle.
 */
template <typename T>
struct OptionalRef {
    T* m_ptr = nullptr;

    OptionalRef() = default;

    explicit constexpr OptionalRef(T& value) noexcept
        : m_ptr(std::addressof(value)) {
    }

    explicit constexpr OptionalRef(T&&) = delete;

    [[nodiscard]]
    constexpr bool has_value() const noexcept {
        return m_ptr != nullptr;
    }

    [[nodiscard]]
    constexpr T& get() const noexcept {
        KOKKOS_EXPECTS(has_value());
        return *m_ptr;
    }
};

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_OPTIONAL_REF_HPP
