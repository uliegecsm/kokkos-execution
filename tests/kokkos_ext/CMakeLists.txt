if(CUDA IN_LIST Kokkos_DEVICES)
    set(LAUNCHER_ARGS LAUNCHER compute-sanitizer --leak-check=full)
else()
    set(LAUNCHER_ARGS)
endif()

add_one_test(NAME parallel_for    ${LAUNCHER_ARGS})
add_one_test(NAME parallel_reduce ${LAUNCHER_ARGS})
