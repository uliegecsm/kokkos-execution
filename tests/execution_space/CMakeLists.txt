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

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION STREQUAL 14.2.0)
  foreach(DEVICE IN LISTS Kokkos_DEVICES)
    set(target_name tests_execution_space_any_sender.${DEVICE})
    set_target_properties(${target_name} PROPERTIES EXCLUDE_FROM_ALL TRUE EXCLUDE_FROM_DEFAULT_BUILD TRUE)
    set_tests_properties(${target_name} PROPERTIES DISABLED TRUE)
    add_test(NAME ${target_name}.compiler COMMAND ${CMAKE_COMMAND} --build . --config $<CONFIGURATION> --target
                                                  ${target_name} --verbose WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    set_tests_properties(
      ${target_name}.compiler
      PROPERTIES PASS_REGULAR_EXPRESSION
                 "error: use of deleted function \'Kokkos::Execution::Impl::Immovable::Immovable"
    )
  endforeach()
endif()
