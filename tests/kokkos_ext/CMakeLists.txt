# Add helper files to the documentation.
set_property(GLOBAL APPEND PROPERTY GraphDispatching_FILES_FOR_DOC ${CMAKE_CURRENT_SOURCE_DIR}/Helpers.hpp)

if(CUDA IN_LIST Kokkos_DEVICES)
    set(LAUNCHER_ARGS LAUNCHER compute-sanitizer --leak-check=full)
else()
    set(LAUNCHER_ARGS)
endif()

add_one_test(NAME exec_scheduler  ${LAUNCHER_ARGS})
add_one_test(NAME graph_scheduler ${LAUNCHER_ARGS})
add_one_test(NAME parallel_for    ${LAUNCHER_ARGS})
add_one_test(NAME parallel_reduce ${LAUNCHER_ARGS})
add_one_test(NAME utils           ${LAUNCHER_ARGS})
