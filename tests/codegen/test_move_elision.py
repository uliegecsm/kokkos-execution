import functools
import json
import logging
import os
import pathlib
import shlex
import subprocess
import typing

import cmake_file_api.kinds
import cmake_file_api.kinds.toolchains.v1
import cmake_file_api.reply.v1
import ijson
import pytest
import regex

# ADD PYTLINT/RUFF/... and others to the pre-commit!

CacheDict: typing.TypeAlias = dict[str, dict[str, typing.Any]]
"""
The CMake cache is a nested dictionary.

References:

* https://cmake.org/cmake/help/latest/manual/cmake-file-api.7.html#object-kind-cache
"""

ToolchainDict: typing.TypeAlias = dict[str, dict[str, typing.Any]]
"""
The CMake toolchain is a nested dictionary.

References:

* https://cmake.org/cmake/help/latest/manual/cmake-file-api.7.html#object-kind-toolchains
"""

CodemodelDict: typing.TypeAlias = dict[str, str | list[typing.Any]]
"""
The CMake code model is a dictionary.

References:

* https://cmake.org/cmake/help/latest/manual/cmake-file-api.7.html#object-kind-codemodel
"""

TargetDict: typing.TypeAlias = dict[str, typing.Any]
"""
The CMake target is a dictionary.

References:

* https://cmake.org/cmake/help/latest/manual/cmake-file-api.7.html#codemodel-version-2-target-object
"""

class FileAPI:
    """
    Retrieve information about the CMake buildsystem.

    References:

    * https://cmake.org/cmake/help/latest/manual/cmake-file-api.7.html
    * https://github.com/madebr/python-cmake-file-api.
    """
    CACHE_VERSION = 2
    TOOLCHAINS_VERSION = 1
    CODEMODEL_VERSION = 2

    def __init__(self, cmake_build_directory: pathlib.Path) -> None:
        self.reader = cmake_file_api.reply.v1.api.CMakeFileApiV1(build_path=cmake_build_directory)
        self.cmake_reply_path = cmake_build_directory / '.cmake' / 'api' / 'v1' / 'reply'

    @functools.cached_property
    def cache(self) -> CacheDict:
        """
        Retrieve the CMake cache.
        """
        cache_json = self.cmake_reply_path / self.reader.index().reply.stateless[cmake_file_api.kinds.kind.ObjectKind.CACHE, self.CACHE_VERSION].jsonFile

        with cache_json.open('rb') as file:
            entries = ijson.items(file, 'entries.item')
            return {
                entry['name']: {k: v for k, v in entry.items() if k != 'name'}
                for entry in entries
            }

    @functools.cached_property
    def toolchains(self) -> ToolchainDict:
        """
        Retrieve the toolchains information.
        """
        toolchains_json = self.cmake_reply_path / self.reader.index().reply.stateless[cmake_file_api.kinds.kind.ObjectKind.TOOLCHAINS, self.TOOLCHAINS_VERSION].jsonFile

        with toolchains_json.open('rb') as file:
            toolchains = ijson.items(file, 'toolchains.item')
            return {
                toolchain['language']: {k: v for k, v in toolchain.items() if k != 'language'}
                for toolchain in toolchains
            }

    @functools.cached_property
    def codemodel_configuration(self) -> CodemodelDict:
        """
        Retrieve the codemodel information, and extract the information available for the build configuration.

        This function assumes that there is only a single build configuration.
        """
        codemodel_json = self.cmake_reply_path / self.reader.index().reply.stateless[cmake_file_api.kinds.kind.ObjectKind.CODEMODEL, self.CODEMODEL_VERSION].jsonFile

        with codemodel_json.open('rb') as file:
            configurations = list(ijson.items(file, 'configurations.item'))

        assert len(configurations) == 1, "Handling of multiple CMake configurations not implemented."

        return configurations[0]

    @functools.lru_cache(maxsize=128) # ruff:ignore[cached-instance-method]
    def target(self, name: str) -> TargetDict:
        """
        Retrieve the information available for the target `name`.
        """
        for target in typing.cast(list[typing.Any], self.codemodel_configuration['targets']):
            if target['name'] == name:
                with (self.cmake_reply_path / target['jsonFile']).open() as file:
                    return json.load(file)
        raise ValueError(f'Target {name!r} not found.')

    @functools.lru_cache(maxsize=128) # ruff:ignore[cached-instance-method]
    def compiler(self, toolchain: str) -> cmake_file_api.kinds.toolchains.v1.CMakeToolchainCompiler:
        """
        Retrieve the compiler for a given toolchain.
        """
        return cmake_file_api.kinds.toolchains.v1.CMakeToolchainCompiler.from_dict(self.toolchains[toolchain]['compiler'])

class TestMoveElision:
    @pytest.fixture(scope='class')
    def cmake_file_api(self) -> FileAPI:
        return FileAPI(cmake_build_directory=pathlib.Path(os.environ["CMAKE_BINARY_DIR"]))

    @pytest.fixture(scope='session')
    def compile_commands(self) -> dict:
        with open(pathlib.Path(os.environ['CMAKE_BINARY_DIR']) / 'compile_commands.json', encoding='utf-8') as fin:
            return json.load(fin)

    def collect(self, name: str, cmake_file_api: FileAPI, compile_commands: dict) -> str:
        """
        Run the compile command, with::

            -Rpass-missed=inline -Rpass=inline
        """
        target = cmake_file_api.target(name=name)
        logging.info(target)

        path = target['artifacts'][0]['path']

        logging.info(path)

        command = [x for x in compile_commands if x['output'] == path]
        assert len(command) == 1

        logging.info(command[0])

        launcher = cmake_file_api.cache['CMAKE_CXX_COMPILER_LAUNCHER']['value']

        cmd = shlex.split(s=launcher) + shlex.split(s=command[0]['command']) + ['-Rpass-missed=inline', '-Rpass=inline']

        output: pathlib.Path = pathlib.Path(os.environ['CMAKE_CURRENT_BINARY_DIR']) / f'{name}.log'

        logging.info(f'Compiling target {name} with:\n\t{cmd}\nand dumping into:\n\t{output}')

        with open(output, 'w+') as out:
            subprocess.check_call(args=cmd, cwd=command[0]['directory'], stdout=out, stderr=subprocess.STDOUT)

        # We are looking for:
        #   * 'not inlined into'
        #   * 'inlined into'
        # A typical line looks like:
        #   /opt/.../Kokkos_SharedAlloc.hpp:632:30: remark: '__clang_call_terminate' not inlined into 'mangled-symbol-A' because it should never be inlined (cost=never): noinline function attribute [-Rpass-missed=inline]
        PATTERN: typing.Final[regex.Pattern] = r'[a-zA-Z0-9._/]+: remark: \'(.*)\' not inlined into \'([a-zA-Z0-9._$]+)'
        with open(output) as out:
            for line in out:
                if 'not inlined into' in line:
                    line = line.lstrip()
                    print(line)
                    if (matched := regex.search(pattern=PATTERN, string=line)) is not None:
                        print("HOURRAY")
                        print(matched.groups())
                        for symbol in matched.groups():
                            print(subprocess.check_output(args=('llvm-cxxfilt', '--types', symbol)).decode().strip())

    def test(self, cmake_file_api: FileAPI, compile_commands: dict) -> None:
        cxx_compiler = cmake_file_api.toolchains['CXX']['compiler']

        logging.info(f'CXX compiler is:\n\t{cxx_compiler}')

        match cxx_compiler['id']:
            case 'Clang':
                self.collect(name='tests_codegen_move_elision.4', cmake_file_api=cmake_file_api, compile_commands=compile_commands)
            case _:
                raise ValueError(cxx_compiler)
