import argparse
import json
import logging
import pathlib

import typeguard

@typeguard.typechecked
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--kokkos-backends', type=str, required=True)
    parser.add_argument('--compiler-family', type=str, required=True)
    parser.add_argument('--dependencies', type=pathlib.Path, required=True)
    parser.add_argument('--input', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    return parser.parse_args()

@typeguard.typechecked
def get_cxx_compiler_for(family: str) -> str:
    match family:
        case 'gnu':
            return 'g++'
        case 'clang':
            return 'clang++'
        case _:
            raise ValueError

@typeguard.typechecked
def install_hpx_requirements(*, compiler_family: str, dependencies: dict) -> str:
    """
    Requirements for the `HPX` backend.
    """
    hpx_ref = dependencies['github']['hpx']
    logging.info(f'Installing requirements for backend HPX (ref {hpx_ref}).')

    cmake_cxx_compiler = get_cxx_compiler_for(family=compiler_family)

    return f"""
ARG HPX_REF={hpx_ref}

RUN --mount=type=tmpfs,target=/tmp/build <<EOF
    set -ex

    apt-helpers install-packages --update --clean --packages hwloc libasio-dev libboost-all-dev

    cd /tmp/build/

    wget https://github.com/STEllAR-GROUP/hpx/archive/refs/tags/v${{HPX_REF}}.tar.gz

    tar -xvf v${{HPX_REF}}.tar.gz

    cd hpx-${{HPX_REF}}/

    cmake -S . -B build \\
        -DCMAKE_BUILD_TYPE=Release \\
        -DCMAKE_C_COMPILER=should-not-be-needed \\
        -DCMAKE_CXX_COMPILER={cmake_cxx_compiler} \\
        -DCMAKE_CXX_EXTENSIONS=OFF \\
        -DHPX_WITH_CXX_STANDARD=20 \\
        -DHPX_WITH_EXAMPLES=OFF \\
        -DHPX_WITH_MALLOC=system \\
        -DHPX_WITH_NETWORKING=OFF \\
        -DHPX_WITH_TESTS=OFF \\
        -DHPX_WITH_UNITY_BUILD=OFF

    cmake --build build -j4
    cmake --install build --prefix=/opt/hpx-${{HPX_REF}}
EOF

ENV HPX_ROOT=/opt/hpx-${{HPX_REF}}
"""

@typeguard.typechecked
def main(*, dependencies: pathlib.Path, kokkos_backends: str, input: pathlib.Path, output: pathlib.Path, compiler_family: str) -> None:
    with open(dependencies, 'r') as f:
        deps = json.load(f)

    content: list[str] = []

    for x in kokkos_backends.split(','):
        match x:
            case 'HPX':
                content.append(install_hpx_requirements(compiler_family=compiler_family, dependencies=deps))
            case _:
                logging.info(f'There is no requirement yet for Kokkos backend {x}.')

    output.write_text(input.read_text().replace('REPLACE_REQUIREMENTS_BACKEND', '\n'.join(content)))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(**vars(parse_args()))
