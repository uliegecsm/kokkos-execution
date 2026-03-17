#ifndef KOKKOS_EXECUTION_TESTS_UTILS_CALLBACK_MATCHERS_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_CALLBACK_MATCHERS_HPP

#include <regex>
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

//! Event record message, regex-ready.
static constexpr char event_record_regex[] = ": event (0x[0-9a-f]+) recorded on ";

// NOLINTBEGIN(cppcoreguidelines-macro-usage)
#define MATCHER_FOR_EVENT_RECORD(_exec_)                                                                               \
    AProfileEvent(                                                                                                     \
        ::testing::Field(                                                                                              \
            &Kokkos::utils::callbacks::ProfileEvent::name,                                                             \
            ::testing::MatchesRegex(                                                                                   \
                std::string(Kokkos::Impl::TypeInfo<std::remove_cvref_t<decltype(_exec_)>>::name())                     \
                    .append(event_record_regex)                                                                        \
                    .append(std::to_string(Kokkos::Tools::Experimental::device_id(_exec_))))))

#define MATCHER_FOR_EVENT_WAIT(_exec_, _id_)                                                                           \
    AProfileEventWithName(                                                                                             \
        std::string(Kokkos::Impl::TypeInfo<_exec_>::name()).append(": waiting for event ").append(_id_))
// NOLINTEND(cppcoreguidelines-macro-usage)

//! Extract the event ID based on @ref event_record_regex.
template <typename... RecordedTypes>
std::string extract_event_record_id(const std::variant<RecordedTypes...>& record) {
    const auto& name = std::get<Kokkos::utils::callbacks::ProfileEvent>(record).name;
    std::smatch match;
    if (std::regex_search(name, match, std::regex(event_record_regex))) {
        return match[1].str(); // NOLINT(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
    }
    throw std::runtime_error(std::format("Could not find '{}' in '{}'.", event_record_regex, name));
}

#endif // KOKKOS_EXECUTION_TESTS_UTILS_CALLBACK_MATCHERS_HPP
