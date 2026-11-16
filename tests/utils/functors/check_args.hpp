#ifndef KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_CHECK_ARGS_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_CHECK_ARGS_HPP

namespace Tests::Utils::Functors {

//! Check call operator argument types before delegating to @ref m_functor.
template <typename T, typename... ExptArgs>
struct CheckArgs {
    T m_functor;

    template <typename... Args>
    KOKKOS_FUNCTION void operator()(Args&&... args) const noexcept(noexcept(m_functor(std::forward<Args>(args)...))) {
        static_assert((std::same_as<Args, ExptArgs> && ...));
        m_functor(std::forward<Args>(args)...);
    }
};

} // namespace Tests::Utils::Functors

#endif // KOKKOS_EXECUTION_TESTS_UTILS_FUNCTORS_CHECK_ARGS_HPP
