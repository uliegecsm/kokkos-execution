#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_DOMAIN_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_DOMAIN_HPP

#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wswitch-default")
#include <stdexec/execution.hpp>
PRAGMA_DIAGNOSTIC_POP

#if defined(KOKKOS_EXECUTION_DEBUG)
#    include "plog/Log.h"
#endif

#include "kokkos-execution/execution_space/context_fwd.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

struct Domain : public stdexec::default_domain {
    template <typename Tag, stdexec::sender Sndr, typename... Args>
    requires stdexec::__callable<ApplySenderFor<Tag>, Sndr, Args...>
    static auto apply_sender(Tag, Sndr&& sndr, Args&&... args) {
#if defined(KOKKOS_EXECUTION_DEBUG)
        PLOG_DEBUG << Kokkos::Impl::TypeInfo<Domain>::name() << ": apply_sender for tag "
                   << Kokkos::Impl::TypeInfo<Tag>::name();
#endif
        return ApplySenderFor<Tag>{}(std::forward<Sndr>(sndr), std::forward<Args>(args)...);
    }

    template <stdexec::sender Sndr, typename Env>
    requires stdexec::__applicable<TransformSenderFor<stdexec::tag_of_t<Sndr>>, Sndr&&, const Env&>
    static auto transform_sender(stdexec::set_value_t, Sndr&& sndr, const Env& env)
        noexcept(stdexec::__nothrow_applicable<TransformSenderFor<stdexec::tag_of_t<Sndr>>, Sndr&&, const Env&>) {
#if defined(KOKKOS_EXECUTION_DEBUG)
        PLOG_DEBUG << Kokkos::Impl::TypeInfo<Domain>::name() << ": transform_sender for tag "
                   << Kokkos::Impl::TypeInfo<stdexec::tag_of_t<Sndr>>::name();
#endif
        return stdexec::__apply(TransformSenderFor<stdexec::tag_of_t<Sndr>>{}, std::forward<Sndr>(sndr), env);
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_DOMAIN_HPP
