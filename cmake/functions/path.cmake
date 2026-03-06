# Get a representation of the relative path from a directory to the parent path of a file.
#
# In this representation, the ``/`` are replaced with ``_``.
function(get_parent_relpath_repr)

  cmake_parse_arguments(gprr "" "DIRECTORY;FILENAME" "" ${ARGN})

  cmake_path(GET gprr_FILENAME PARENT_PATH FILE_PARENT_PATH)
  cmake_path(RELATIVE_PATH FILE_PARENT_PATH BASE_DIRECTORY "${gprr_DIRECTORY}" OUTPUT_VARIABLE PARENT_RELPATH)

  string(REPLACE "\/" "_" PARENT_RELPATH_REPR "${PARENT_RELPATH}")

  set(gprr_PARENT_RELPATH_REPR "${PARENT_RELPATH_REPR}" PARENT_SCOPE)

endfunction(get_parent_relpath_repr)
