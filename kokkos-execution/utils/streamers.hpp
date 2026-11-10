#ifndef KOKKOS_EXECUTION_UTILS_STREAMERS_HPP
#define KOKKOS_EXECUTION_UTILS_STREAMERS_HPP

#include <concepts>
#include <ostream>
#include <ranges>
#include <vector>

namespace Kokkos::Execution::Utils {

namespace detail {

template <typename T>
concept char_like = std::same_as<std::remove_cv_t<T>, char> || std::same_as<std::remove_cv_t<T>, wchar_t>
                 || std::same_as<std::remove_cv_t<T>, char8_t> || std::same_as<std::remove_cv_t<T>, char16_t>
                 || std::same_as<std::remove_cv_t<T>, char32_t>;

//! A (const-iterable) range of streamable, non-character elements.
template <class R>
concept streamable_range = std::ranges::input_range<const R> && !char_like<std::ranges::range_value_t<const R>>
                        && requires(std::ostream& out, std::ranges::range_reference_t<const R> value) { out << value; };
} // namespace detail

template <detail::streamable_range R>
std::ostream& operator<<(std::ostream& out, const R& values) {
    out << '[';
    for (bool first = true; const auto& value: values) {
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
