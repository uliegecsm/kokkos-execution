#ifndef KOKKOS_EXECUTION_GRAPH_DOMAIN_HPP
#define KOKKOS_EXECUTION_GRAPH_DOMAIN_HPP

#include "kokkos-execution/stdexec.hpp"

namespace Kokkos::Execution::GraphImpl {

struct Domain : public stdexec::default_domain { };

} // namespace Kokkos::Execution::GraphImpl

#endif // KOKKOS_EXECUTION_GRAPH_DOMAIN_HPP
