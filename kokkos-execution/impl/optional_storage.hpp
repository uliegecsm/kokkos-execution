#ifndef KOKKOS_EXECUTION_IMPL_OPTIONAL_STORAGE_HPP
#define KOKKOS_EXECUTION_IMPL_OPTIONAL_STORAGE_HPP

#include "kokkos-execution/stdexec.hpp"

#include "Kokkos_Core.hpp"

namespace Kokkos::Execution::Impl {

/**
 * @brief Data structure that has an interface similar to @c std::optional but stores the value with @c stdexec::__manual_lifetime.
 *
 * 1. Similarly to @c std::optional, the boolean member @ref m_has_value tracks if the storage has been populated or not.
 * 2. Similarly to @c std::optional, it will destroy the underlying storage automatically in the destructor if @ref m_has_value is @c true.
 * 3. It provides similar storage semantics as @c stdexec::__manual_lifetime: it is not movable, and can be used as a placeholder storage for types that are not movable, like @c stdexec operation states.
 *    Notably, the member function @ref emplace_from allows in-place construction from the result of a callable.
 */
template <typename T>
class OptionalStorage {
   public:
    constexpr OptionalStorage() noexcept = default;

    constexpr ~OptionalStorage() {
        if (m_has_value) {
            m_storage.__destroy();
        }
    }

    OptionalStorage(const OptionalStorage&) = delete;
    OptionalStorage(OptionalStorage&&) = delete;
    OptionalStorage& operator=(const OptionalStorage&) = delete;
    OptionalStorage& operator=(OptionalStorage&&) = delete;

    template <typename... Args>
    constexpr auto emplace(Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args...>) -> T& {
        KOKKOS_ASSERT(!m_has_value);
        auto& ref = m_storage.__construct(std::forward<Args>(args)...);
        m_has_value = true;
        return ref;
    }

    template <typename Func, typename... Args>
    constexpr auto emplace_from(Func&& func, Args&&... args)
        noexcept(noexcept(T{std::forward<Func>(func)(std::forward<Args>(args)...)})) -> T& {
        KOKKOS_ASSERT(!m_has_value);
        auto& ref = m_storage.__construct_from(std::forward<Func>(func), std::forward<Args>(args)...);
        m_has_value = true;
        return ref;
    }

    constexpr void reset() noexcept {
        KOKKOS_ASSERT(m_has_value);
        m_storage.__destroy();
        m_has_value = false;
    }

    constexpr auto get() & noexcept -> T& {
        KOKKOS_ASSERT(m_has_value);
        return m_storage.__get();
    }

    constexpr auto get() const & noexcept -> const T& {
        KOKKOS_ASSERT(m_has_value);
        return m_storage.__get();
    }

    constexpr auto get() && noexcept -> T&& {
        KOKKOS_ASSERT(m_has_value);
        return std::move(m_storage.__get());
    }

    constexpr auto get() const && noexcept -> const T&& = delete;

    constexpr auto operator->() noexcept -> T* {
        KOKKOS_ASSERT(m_has_value);
        return m_storage.operator->();
    }

    constexpr auto operator->() const noexcept -> const T* {
        KOKKOS_ASSERT(m_has_value);
        return m_storage.operator->();
    }

    constexpr explicit operator bool() const noexcept {
        return m_has_value;
    }

    constexpr bool has_value() const noexcept {
        return m_has_value;
    }

   private:
    stdexec::__manual_lifetime<T> m_storage{};
    bool m_has_value{false};
};

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_OPTIONAL_STORAGE_HPP
