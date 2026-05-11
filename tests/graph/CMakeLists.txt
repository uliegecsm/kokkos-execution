add_one_test(NAME bulk)
add_one_test(NAME continues_on)
add_one_test(NAME diagnostics.failure)
add_one_test(NAME domain)
add_one_test(NAME events)
add_one_test(NAME fork_join)
add_one_test(NAME node_props)
add_one_test(NAME operation_state)
add_one_test(NAME operation_state.failure)
add_one_test(NAME parallel_for)
add_one_test(NAME parallel_reduce)
add_one_test(NAME repeat_until)
add_one_test(NAME scheduler)
add_one_test(NAME sync_wait)
add_one_test(NAME then)
add_one_test(NAME when_all)
add_one_test(NAME when_all.failure)

foreach(DEVICE IN LISTS Kokkos_DEVICES)
  set(target_name tests_graph_when_all.failure.${DEVICE})
  add_build_failure_test(
    ${target_name}.compiler ${target_name}
    "CANNOT_DISPATCH_THIS_ALGORITHM_TO_THE_GRAPH_SCHEDULER.*BECAUSE_THERE_IS_NO_GRAPH_SCHEDULER_IN_THE_ENVIRONMENT"
  )

  set(target_name tests_graph_diagnostics.failure.${DEVICE})
  add_build_failure_test(${target_name}.compiler ${target_name} "Check for the error in the compiler log.")
  set(target_name tests_graph_operation_state.failure.${DEVICE})
  add_build_failure_test(
    ${target_name}.compiler ${target_name} "This operation state that not satisfy 'RemainsOnGraphFor'"
  )
endforeach()
