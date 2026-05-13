#ifndef KOKKOS_EXECUTION_IMPL_CONTINUATION_PROP_HPP
#define KOKKOS_EXECUTION_IMPL_CONTINUATION_PROP_HPP

#include "Kokkos_Core.hpp"

#include "kokkos-execution/impl/event.hpp"
#include "kokkos-execution/impl/get_exec.hpp"

namespace Kokkos::Execution::Impl {

template <Kokkos::ExecutionSpace Exec>
struct OrderedOn {
    ExecutionSpaceRef<Exec> m_exec_ref;

    constexpr explicit OrderedOn(const Exec& exec) noexcept
        : m_exec_ref(exec) {
    }

    constexpr auto exec() const & noexcept -> const Exec& {
        return m_exec_ref.get();
    }
};

template <typename EventType>
struct DependsOn {
    EventType const * m_event;

    constexpr explicit DependsOn(const EventType& event) noexcept
        : m_event(std::addressof(event)) {
    }

    constexpr auto event() const & noexcept -> const EventType& {
        return *m_event;
    }
};

template <typename Prop>
struct IsOrderedOnExec : std::false_type { };

template <Kokkos::ExecutionSpace Exec>
struct IsOrderedOnExec<OrderedOn<Exec>> : std::true_type { };

template <typename Prop>
concept is_ordered_on_exec = IsOrderedOnExec<Prop>::value;

template <typename Prop>
struct IsDependsOnEvent : std::false_type { };

template <Kokkos::ExecutionSpace Exec>
struct IsDependsOnEvent<DependsOn<Event<Exec>>> : std::true_type { };

template <typename Prop>
concept is_depends_on_event = IsDependsOnEvent<Prop>::value;

template <typename Prop>
concept is_continuation_property = is_ordered_on_exec<Prop> || is_depends_on_event<Prop>;

template <typename Prop>
requires(is_continuation_property<Prop>)
struct ContinuationProp {
    Prop m_prop;

    constexpr explicit ContinuationProp(const Prop& prop) noexcept
        : m_prop(prop) {
    }

    constexpr auto get() const & noexcept -> const Prop& {
        return m_prop;
    }
};

template <typename Prop>
ContinuationProp(Prop) -> ContinuationProp<std::remove_cvref_t<Prop>>;

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_CONTINUATION_PROP_HPP
