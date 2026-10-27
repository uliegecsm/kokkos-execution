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

#if defined(__cpp_impl_reflection)

#    include <meta>

namespace Kokkos::Execution::Utils {

//! @name Annotation to mark a type as streamable.
///@{
struct Streamable { };

inline constexpr Streamable streamable{};
///@}

//! Check whether @p type carries the @c [[=streamable]] annotation.
consteval bool is_streamable(std::meta::info type) {
    return !std::meta::annotations_of_with_type(type, ^^Streamable).empty();
}

//! Constrain a type @p T that opted in to the reflection-generated @c operator<<.
template <typename T>
concept IsStreamable = is_streamable(^^T);

template <typename T>
std::ostream& stream_to(std::ostream& out, const T& obj) {
    static constexpr auto members = std::define_static_array(
        std::meta::nonstatic_data_members_of(^^T, std::meta::access_context::current()));

    out << std::meta::identifier_of(^^T) << ": {";

    template for (constexpr std::meta::info member: members) {
        if constexpr (member != members.front()) {
            out << ", ";
        }
        out << std::meta::identifier_of(member) << " = " << obj.[:member:];
    }

    return out << '}';
}

//! Reflection-generated @c operator<< streaming "TypeName: {member = value, member = value}".
template <IsStreamable T>
std::ostream& operator<<(std::ostream& out, const T& obj) {
    return stream_to(out, obj);
}

} // namespace Kokkos::Execution::Utils

#    define KOKKOS_EXECUTION_UTILS_STREAMABLE [[= Kokkos::Execution::Utils::streamable]]
#else
#    define KOKKOS_EXECUTION_UTILS_STREAMABLE
#endif

#endif // KOKKOS_EXECUTION_UTILS_STREAMERS_HPP
