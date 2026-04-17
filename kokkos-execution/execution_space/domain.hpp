#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_DOMAIN_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_DOMAIN_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/execution_space/execution_space_fwd.hpp"
#include "kokkos-execution/impl/domain.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

struct Domain
    : public Impl::ApplySender<Domain, ApplySenderFor>
    , public Impl::TransformSender<Domain, TransformSenderFor> {
    using Impl::ApplySender<Domain, ApplySenderFor>::apply_sender;
    using Impl::TransformSender<Domain, TransformSenderFor>::transform_sender;
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_DOMAIN_HPP
