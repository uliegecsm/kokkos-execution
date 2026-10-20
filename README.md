[![Run tests](https://github.com/uliegecsm/kokkos-execution/actions/workflows/test.yml/badge.svg)](https://github.com/uliegecsm/kokkos-execution/actions/workflows/test.yml)

# kokkos-execution

`kokkos-execution` is a header-only library
that customizes `std::execution` for [`Kokkos`](https://github.com/kokkos/kokkos/).
It uses [`stdexec`](https://github.com/NVIDIA/stdexec/), the reference implementation of `std::execution`.

`kokkos-execution` lets you express your operations as sender pipelines,
and lower them transparently to either of these two `Kokkos` execution models:
1. [execution space](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Machine-Model.html#execution-space-instances)
2. [graph](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Graph.html)

A thorough comparison of these execution models can be found [here](https://orbi.uliege.be/handle/2268/348774).

The full documentation is available [here](https://uliegecsm.github.io/kokkos-execution/).

## Overview

`kokkos-execution` provides the following execution contexts:
1. [`Kokkos::Execution::ExecutionSpaceContext`](kokkos-execution/execution_space.hpp)
2. [`Kokkos::Execution::GraphContext`](kokkos-execution/graph.hpp)

Many standard algorithms (such as `then`, `bulk`, `continues_on`, `when_all`)
are customized for each of these contexts,
together with the custom algorithms
[`Kokkos::Execution::parallel_for`](kokkos-execution/parallel_for.hpp)
[`Kokkos::Execution::parallel_reduce`](kokkos-execution/parallel_reduce.hpp).

The algorithms are customized to avoid unnecessary synchronization points.

All of the following `Kokkos` backends are supported and exercised in CI:

| `Kokkos` backend | Supported          |
|------------------|--------------------|
| `Serial`         | :white_check_mark: |
| `Threads`        | :exclamation:      |
| `OpenMP`         | :white_check_mark: |
| `HPX`            | :white_check_mark: |
| `Cuda`           | :white_check_mark: |
| `HIP`            | :white_check_mark: |
| `SYCL`           | :white_check_mark: |

> :exclamation: The `Kokkos::Threads` backend requires that parallel regions be launched from the thread that initialized `Kokkos`.
> Many tests are thus disabled.

### How operation dependencies are expressed?

When using `Kokkos::Execution::ExecutionSpaceContext`, dependencies are expressed either
through execution space ordering or using [events](kokkos-execution/impl/event.hpp).

When using `Kokkos::Execution::GraphContext`, dependencies are expressed
through the underlying `Kokkos` graph edges.

### Testing

Tests do not rely solely on value correctness.
Through the [`kokkos-tools`](https://github.com/kokkos/kokkos-tools) hooks,
they also assert that the expected sequence of `Kokkos` actions (such as parallel region launch or fence)
actually took place.
Such trace-based assertions are made practical by [`kokkos-utils`](https://github.com/uliegecsm/kokkos-utils),
which provides the matcher machinery used to express and check the expected sequences.

Tests can be found in [this directory](tests/).

## Building

The library is developed against the latest version of `stdexec`, bundled as a submodule.
The versions of `Kokkos` and other dependencies are [recorded here](dependencies.json).

Our CI relies on Docker images, ready to be used for testing `kokkos-execution` off the self:
```bash
docker pull ghcr.io/uliegecsm/kokkos-execution/kokkos-execution:gnu-15-OpenMP-Release
```

Once the container is running, do:
```bash
git clone --recurse-submodules https://github.com/uliegecsm/kokkos-execution.git
cd kokkos-execution
cmake -S . --preset=gnu-OpenMP
cmake --build --preset=gnu-OpenMP
ctest --preset=gnu-OpenMP
```

## Example

A sequence of operations can be written as a sender pipeline
and easily lowered to either an execution space instance or a graph:
```c++
auto ops = Kokkos::Execution::parallel_for("label for A", 17, Functor_A{...})
         | stdexec::bulk(stdexec::par, N_B, Functor_B{...})
         | Kokkos::Execution::parallel_for("label for C", Kokkos::range_slice<size_t, size_t>{3, 42}, Functor_C{...});

// Lower onto an execution space instance.
const Kokkos::Execution::ExecutionSpaceContext<Exec> ectx{exec};
stdexec::sync_wait(stdexec::schedule(ectx.get_scheduler()) | ops);

// Lower onto a graph.
const Kokkos::Execution::GraphContext<Exec> gctx{exec};
stdexec::sync_wait(stdexec::schedule(gctx.get_scheduler()) | ops);
```

More examples are available [here](examples/).

## Contributing

See our [contribution guidelines](CONTRIBUTING.md).
