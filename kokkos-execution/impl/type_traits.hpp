#ifndef KOKKOS_EXECUTION_IMPL_TYPE_TRAITS_HPP
#define KOKKOS_EXECUTION_IMPL_TYPE_TRAITS_HPP

#include <type_traits>
#include <utility>

#include "Kokkos_Macros.hpp"

namespace Kokkos::Execution::Impl {

/**
 * @brief Yields the type of member @p _member_ as accessed from expression @p _Self_,
 *        preserving the @c cv and @c ref qualifiers of @p _Self_.
 *
 * Useful for propagating value category from a deducing-this parameter to one of its members.
 */ // NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KOKKOS_EXECUTION_IMPL_MEMBER_CVREF_T(_Self_, _member_) decltype((std::declval<_Self_>()._member_))

/**
 * @brief Equivalent to @c std::forward<_Self_>(_self_).
 *
 * Avoids @c clang-tidy @c bugprone-use-after-move heuristic issues.
 */ // NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KOKKOS_EXECUTION_IMPL_FORWARD_THIS(_Self_, _self_)     static_cast<_Self_&&>(_self_)

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_TYPE_TRAITS_HPP
