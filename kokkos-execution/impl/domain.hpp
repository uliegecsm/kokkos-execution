#ifndef KOKKOS_EXECUTION_IMPL_DOMAIN_HPP
#define KOKKOS_EXECUTION_IMPL_DOMAIN_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/impl/type_info.hpp"

#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
#    include "plog/Log.h"
#endif

namespace Kokkos::Execution::Impl {

template <typename Domain, template <typename> typename ApplySenderForType>
struct ApplySender {
    template <typename Tag, stdexec::sender Sndr, typename... Args>
    requires stdexec::__callable<ApplySenderForType<Tag>, Sndr&&, Args&&...>
    static auto apply_sender(Tag, Sndr&& sndr, Args&&... args)
        noexcept(stdexec::__nothrow_callable<ApplySenderForType<Tag>, Sndr&&, Args&&...>) {
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
        PLOG_DEBUG << Impl::display_string_of<Domain>() << ": apply_sender for tag " << Impl::display_string_of<Tag>();
#endif
        return ApplySenderForType<Tag>{}(std::forward<Sndr>(sndr), std::forward<Args>(args)...);
    }
};

template <typename Domain, template <typename> typename TransformSenderForType>
struct TransformSender {
    template <stdexec::sender Sndr, typename Env>
    requires stdexec::__applicable<TransformSenderForType<stdexec::tag_of_t<Sndr>>, Sndr&&, const Env&>
    static auto transform_sender(stdexec::set_value_t, Sndr&& sndr, const Env& env)
        noexcept(stdexec::__nothrow_applicable<TransformSenderForType<stdexec::tag_of_t<Sndr>>, Sndr&&, const Env&>) {
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
        PLOG_DEBUG << Impl::display_string_of<Domain>() << ": transform_sender for tag "
                   << Impl::display_string_of<stdexec::tag_of_t<Sndr>>();
#endif
        return stdexec::__apply(TransformSenderForType<stdexec::tag_of_t<Sndr>>{}, std::forward<Sndr>(sndr), env);
    }
};

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_DOMAIN_HPP
