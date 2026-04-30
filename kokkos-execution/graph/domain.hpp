#ifndef KOKKOS_EXECUTION_GRAPH_DOMAIN_HPP
#define KOKKOS_EXECUTION_GRAPH_DOMAIN_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/graph/graph_fwd.hpp"
#include "kokkos-execution/impl/domain.hpp"

namespace Kokkos::Execution::GraphImpl {

struct Domain
    : public Impl::ApplySender<Domain, ApplySenderFor>
    , public Impl::TransformSender<Domain, TransformSenderFor> {
    using Impl::ApplySender<Domain, ApplySenderFor>::apply_sender;
    using Impl::TransformSender<Domain, TransformSenderFor>::transform_sender;
};

} // namespace Kokkos::Execution::GraphImpl

#endif // KOKKOS_EXECUTION_GRAPH_DOMAIN_HPP
