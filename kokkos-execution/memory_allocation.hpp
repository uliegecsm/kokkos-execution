#ifndef KOKKOS_EXECUTION_MEMORY_ALLOCATION_HPP
#define KOKKOS_EXECUTION_MEMORY_ALLOCATION_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-declarations")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
#include "kokkos-execution/stdexec/P3955/enter_scope_sender.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "Kokkos_Macros.hpp"
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#    include "Kokkos_Core.hpp"
#endif

namespace Kokkos::Execution {

/**
 * When entering the scope, allocation a @c Kokkos view of type @ref view_t.
 * When exiting the scope, deallocate it.
 *
 * While the data is managed by @ref view_t, when used with a @c exec::lifetime, the user receives
 * a @ref type (which is a @c std::mdspan). This is on purpose to avoid that users inadvertently copy the view,
 * which is reference counted, and thus fiddle with the @c exec::lifetime promise.
 *
 * For use cases that really require a @c Kokkos::View, the user can use `exec::sync_object<Kokkos::View<...>>` directly.
 *
 * See also @cite P3955R2.
 *
 * @todo The idea could be extended to a custom algorithm that, under @ref Kokkos::Execution::ExecutionSpaceContext,
 *       enqueues the allocation and deallocation operations on the underlying execution space instance, and that, under
 *       @ref Kokkos::Execution::GraphContext, uses memory nodes.
 */
template <typename ViewType, typename... Args>
struct memory_allocation {
    static_assert(Kokkos::is_view_v<ViewType>);

    using view_t = ViewType;
    using type = typename view_t::mdspan_type;

    template <typename... Ts>
    requires(std::is_constructible_v<Args, Ts> && ...)
    constexpr explicit memory_allocation(Ts&&... args) noexcept((std::is_nothrow_constructible_v<Args, Ts> && ...))
        : m_args(std::forward<Ts>(args)...) {
    }

    template <typename Self>
    constexpr experimental::execution::enter_scope_sender auto
        STDEXEC_EXPLICIT_THIS_BEGIN(operator())(this Self&& self, type* storage) noexcept(
        std::is_nothrow_constructible_v<std::tuple<Args...>, stdexec::__copy_cvref_t<Self, std::tuple<Args...>>>) {
        return stdexec::just(std::forward<Self>(self).m_args)
             | stdexec::then(
                   [storage](std::tuple<Args...>&& args) noexcept(std::is_nothrow_constructible_v<view_t, Args...>) {
                       auto view = std::make_from_tuple<view_t>(std::move(args));
                       // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
                       type* const ptr = ::new (static_cast<void*>(storage)) type(view.to_mdspan());
                       return stdexec::just() | stdexec::then([ptr, view = std::move(view)]() mutable noexcept {
                                  ptr->~type();
                                  view = view_t{};
                              });
                   });
    }
    STDEXEC_EXPLICIT_THIS_END(operator())

   private:
    std::tuple<Args...> m_args;
};

template <typename... ViewArgs, typename... Args>
constexpr memory_allocation<Kokkos::View<ViewArgs...>, std::decay_t<Args>...>
    make_memory_allocation(Args&&... args) noexcept(
        (std::is_nothrow_constructible_v<memory_allocation<Kokkos::View<ViewArgs...>, std::decay_t<Args>...>, Args&&>
         && ...)) {
    return memory_allocation<Kokkos::View<ViewArgs...>, std::decay_t<Args>...>(std::forward<Args>(args)...);
}

} // namespace Kokkos::Execution

#endif // KOKKOS_EXECUTION_MEMORY_ALLOCATION_HPP
