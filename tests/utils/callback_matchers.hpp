#ifndef KOKKOS_EXECUTION_TESTS_UTILS_CALLBACK_MATCHERS_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_CALLBACK_MATCHERS_HPP

#include <string>
#include <string_view>

#include "Kokkos_Core.hpp"

#include "kokkos-utils/callbacks/Helpers.hpp"

/**
 * @file
 *
 * This file provides some helpers for matchers from @ref Kokkos::utils::callbacks.
 *
 * @todo Let's revisit these helpers in light of what happens in https://github.com/uliegecsm/kokkos-utils/issues/42.
 */

// NOLINTBEGIN(cppcoreguidelines-macro-usage)
#define MATCHER_FOR_NAME(_type_, _name_)                                                                               \
    ::testing::Field(&Kokkos::utils::callbacks::_type_##Event::name, ::testing::StrEq(_name_))

#define MATCHER_FOR_DEV_ID(_type_, _exec_)                                                                             \
    ::testing::Field(                                                                                                  \
        &Kokkos::utils::callbacks::_type_##Event::dev_id,                                                              \
        ::testing::Eq(Kokkos::Tools::Experimental::device_id(_exec_)))

#define MATCHER_FOR_BEGIN_FENCE(_exec_, _label_)                                                                       \
    ABeginFenceEvent(MATCHER_FOR_NAME(BeginFence, _label_), MATCHER_FOR_DEV_ID(BeginFence, _exec_))
#define MATCHER_FOR_BEGIN_PFOR(_exec_, _label_)                                                                        \
    ABeginParallelForEvent(MATCHER_FOR_NAME(BeginParallelFor, _label_), MATCHER_FOR_DEV_ID(BeginParallelFor, _exec_))
#define MATCHER_FOR_BEGIN_PRED(_exec_, _label_)                                                                        \
    ABeginParallelReduceEvent(                                                                                         \
        MATCHER_FOR_NAME(BeginParallelReduce, _label_), MATCHER_FOR_DEV_ID(BeginParallelReduce, _exec_))
#define MATCHER_FOR_PUSH_REGION(_label_) APushRegionEventWithName(::testing::StrEq(_label_))
#define MATCHER_FOR_POP_REGION()         APopRegionEvent()

#define MATCHER_FOR_BEGIN_DEEP_COPY(_dst_, _src_)                                                                      \
    ABeginDeepCopyEvent(                                                                                               \
        Kokkos::utils::callbacks::PartialMatcher<Kokkos::utils::callbacks::BeginDeepCopyEvent>{}(                      \
            Kokkos::utils::callbacks::BeginDeepCopyEvent{                                                              \
                .dst = KOKKOS_IMPL_STRIP_PARENS(_dst_), .src = KOKKOS_IMPL_STRIP_PARENS(_src_)}))
// NOLINTEND(cppcoreguidelines-macro-usage)

//! Get the dispatch label from @p Exec and @p label.
template <Kokkos::ExecutionSpace Exec, std::convertible_to<std::string_view> Label>
constexpr std::string dispatch_label(const Exec&, Label&& label) {
    return std::string(Kokkos::Impl::TypeInfo<Exec>::name()).append(": ").append(std::forward<Label>(label));
}

#if defined(KOKKOS_EXECUTION_IMPL_EVENT_HPP)
DEFINE_EVENT_MATCHER_IN(Kokkos::Execution::Impl, RecordEvent)
DEFINE_EVENT_MATCHER_IN(Kokkos::Execution::Impl, WaitEvent)
#endif

// NOLINTBEGIN(cppcoreguidelines-macro-usage)
#define MATCHER_FOR_RECORD_EVENT(_exec_)                                                                               \
    ARecordEvent(                                                                                                      \
        ::testing::Field(                                                                                              \
            &Kokkos::Execution::Impl::RecordEvent::dev_id,                                                             \
            ::testing::Eq(Kokkos::Tools::Experimental::device_id(_exec_))))

#define MATCHER_FOR_WAIT_EVENT(_record_event_variant_)                                                                 \
    AWaitEvent(                                                                                                        \
        ::testing::Field(                                                                                              \
            &Kokkos::Execution::Impl::WaitEvent::event_id,                                                             \
            ::testing::Eq(std::get<Kokkos::Execution::Impl::RecordEvent>(_record_event_variant_).event_id)))
// NOLINTEND(cppcoreguidelines-macro-usage)

//! Matcher to filter out events that are just noise for tests.
template <Kokkos::ExecutionSpace Exec>
struct EventDiscardMatcher {
#if defined(KOKKOS_ENABLE_SYCL)
    //! Filter out any @ref Kokkos::utils::callbacks::BeginFenceEvent induced by https://github.com/kokkos/kokkos/blob/91584fc13aaf09330bc391466dbae0249895291f/core/src/SYCL/Kokkos_SYCL_Instance.hpp#L133.
    bool operator()(const Kokkos::utils::callbacks::BeginFenceEvent& event) const
        requires std::same_as<Exec, Kokkos::SYCL>
    {
        return event.name.find("Kokkos::SYCLInternal::USMObject") == std::string::npos;
    }

    //! Filter out any Kokkos::utils::callbacks::AllocateDataEvent induced by https://github.com/kokkos/kokkos/blob/91584fc13aaf09330bc391466dbae0249895291f/core/src/SYCL/Kokkos_SYCL_Instance.cpp#L306.
    bool operator()(const Kokkos::utils::callbacks::AllocateDataEvent& event) const
        requires std::same_as<Exec, Kokkos::SYCL>
    {
        return event.alloc.name.find("Kokkos::SYCL::USMObject") == std::string::npos;
    }
#endif

#if defined(KOKKOS_ENABLE_HPX)
    /**
     * Even after an HPX instance has been fenced, operation states created internally in HPX holding the enqueued work
     * may still be alive and worker threads created internally in HPX may still touch and destroy them. The belated
     * destruction of instances held in such operation states may lead to spurious fence events later on.
     *
     * See https://github.com/uliegecsm/kokkos-execution/issues/150.
     */
    bool operator()(const Kokkos::utils::callbacks::BeginFenceEvent& event) const
        requires std::same_as<Exec, Kokkos::Experimental::HPX>
    {
        return event.name != "Kokkos::Experimental::HPX: fence on destruction";
    }
#endif

    template <Kokkos::utils::callbacks::Event EventType>
    constexpr bool operator()(const EventType&) const {
        return true;
    }
};

#endif // KOKKOS_EXECUTION_TESTS_UTILS_CALLBACK_MATCHERS_HPP
