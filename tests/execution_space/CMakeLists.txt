add_one_test(NAME any_sender)
add_one_test(NAME bulk)
add_one_test(NAME continues_on)
add_one_test(NAME ensure_started)
add_one_test(NAME fork_join)
add_one_test(NAME inter_op)
add_one_test(NAME let_value)
add_one_test(NAME on)
add_one_test(NAME operation_state)
add_one_test(NAME parallel_for)
add_one_test(NAME repeat_until)
add_one_test(NAME scheduler)
add_one_test(NAME scoped_region)
add_one_test(NAME split ADDITIONAL_LINK_LIBRARIES GTest::gmock)
add_one_test(NAME sync_wait)
add_one_test(NAME then)
add_one_test(NAME transfer_when_all)
add_one_test(NAME when_all)
add_one_test(NAME when_all.failure)

foreach(DEVICE IN LISTS Kokkos_DEVICES)
  set(target_name tests_execution_space_when_all.failure.${DEVICE})
  add_build_failure_test(
    ${target_name}.compiler
    ${target_name}
    "CANNOT_DISPATCH_THIS_ALGORITHM_TO_THE_EXECUTION_SPACE_SCHEDULER.*BECAUSE_THERE_IS_NO_EXECUTION_SPACE_SCHEDULER_IN_THE_ENVIRONMENT"
  )
endforeach()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION STREQUAL 14.2.0)
  foreach(DEVICE IN LISTS Kokkos_DEVICES)
    set(target_name tests_execution_space_any_sender.${DEVICE})
    add_build_failure_test(
      ${target_name}.compiler ${target_name}
      "error: use of deleted function \'Kokkos::Execution::Impl::Immovable::Immovable"
    )
  endforeach()
endif()
