#ifndef KOKKOS_EXECUTION_PATH_HPP
#define KOKKOS_EXECUTION_PATH_HPP

namespace Kokkos::Execution {

struct path_t {
    template <typename... Clsrs>
    constexpr auto operator()(Clsrs&&... clsrs) const /* noexcept */ {
    }
};

namespace Impl {
struct PathSender { };

} // namespace Impl

inline constexpr path_t path{};

} // namespace Kokkos::Execution

#endif // KOKKOS_EXECUTION_PATH_HPP
