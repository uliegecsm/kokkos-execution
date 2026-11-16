#ifndef KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_CHECK_ARGS_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_CHECK_ARGS_HPP

namespace Tests::Utils::Functors {

//! Check call operator argument types before delegating to @ref m_functor.
template <typename T, typename Tag = void, typename... ExptArgs>
struct CheckArgs {
    T m_functor;

    template <typename... Args>
    requires std::is_void_v<Tag>
    KOKKOS_FUNCTION void operator()(Args&&... args) const noexcept(noexcept(m_functor(std::forward<Args>(args)...))) {
        static_assert((std::same_as<Args, ExptArgs> && ...));
        m_functor(std::forward<Args>(args)...);
    }

    template <typename... Args, typename U = Tag>
    requires(!std::is_void_v<U>)
    KOKKOS_FUNCTION void
        operator()(const U& tag, Args&&... args) const noexcept(noexcept(m_functor(tag, std::forward<Args>(args)...))) {
        static_assert((std::same_as<Args, ExptArgs> && ...));
        m_functor(tag, std::forward<Args>(args)...);
    }
};

} // namespace Tests::Utils::Functors

#endif // KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_CHECK_ARGS_HPP
