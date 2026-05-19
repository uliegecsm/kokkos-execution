#ifndef KOKKOS_EXECUTION_UTILS_STREAMERS_HPP
#define KOKKOS_EXECUTION_UTILS_STREAMERS_HPP

#include <ostream>
#include <vector>

namespace Kokkos::Execution::Utils {

inline std::ostream& operator<<(std::ostream& out, const std::vector<void*>& values) {
    out << '[';
    for (bool first = true; const void* value: values) {
        if (!first)
            out << ", ";
        out << value;
        first = false;
    }
    return out << ']';
}

} // namespace Kokkos::Execution::Utils

#endif // KOKKOS_EXECUTION_UTILS_STREAMERS_HPP
