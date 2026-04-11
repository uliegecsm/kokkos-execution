#ifndef KOKKOS_EXECUTION_IMPL_DOMAIN_HPP
#define KOKKOS_EXECUTION_IMPL_DOMAIN_HPP

#include "kokkos-execution/stdexec.hpp"

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
        PLOG_DEBUG << Kokkos::Impl::TypeInfo<Domain>::name() << ": apply_sender for tag "
                   << Kokkos::Impl::TypeInfo<Tag>::name();
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
        PLOG_DEBUG << Kokkos::Impl::TypeInfo<Domain>::name() << ": transform_sender for tag "
                   << Kokkos::Impl::TypeInfo<stdexec::tag_of_t<Sndr>>::name();
#endif
        return stdexec::__apply(TransformSenderForType<stdexec::tag_of_t<Sndr>>{}, std::forward<Sndr>(sndr), env);
    }
};

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_DOMAIN_HPP
