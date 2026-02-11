#!/bin/bash
set -ex

CLANG_FORMAT_EXECUTABLE=clang-format

if [ "$#" -eq 0 ]; then
  CLANG_FORMAT_ARGS=(--dry-run --Werror)
else
  CLANG_FORMAT_ARGS=("$@")
fi

git ls-files *.hpp *.h *.cpp *.c | xargs ${CLANG_FORMAT_EXECUTABLE} "${CLANG_FORMAT_ARGS[@]}"
