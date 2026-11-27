# Check whether object files reference the functions that count the references to a Kokkos allocation.
#
# Usage: cmake -DNM=<nm> -DOBJECTS=<object files> -DEXPECT=<absent|present> -P check_refcount_symbols.cmake
#
# Such a reference is a call that the optimizer could not remove. With EXPECT=absent, the check fails if there is any.
# With EXPECT=present, it fails if there is none.

foreach(REQUIRED IN ITEMS NM OBJECTS EXPECT)
  if(NOT DEFINED ${REQUIRED} OR "${${REQUIRED}}" STREQUAL "")
    message(FATAL_ERROR "${REQUIRED} is not set.")
  endif()
endforeach()

if(NOT EXPECT MATCHES "^(absent|present)$")
  message(FATAL_ERROR "EXPECT must be 'absent' or 'present', not '${EXPECT}'.")
endif()

set(REFCOUNT_REGEX "[^\n]*SharedAllocationRecord<void, void>::(increment|decrement)\\([^\n]*")

foreach(OBJECT IN LISTS OBJECTS)
  execute_process(
    COMMAND "${NM}" --undefined-only --demangle "${OBJECT}"
    OUTPUT_VARIABLE UNDEFINED_SYMBOLS
    ERROR_VARIABLE NM_ERROR
    RESULT_VARIABLE NM_STATUS
  )

  if(NOT NM_STATUS EQUAL 0)
    message(FATAL_ERROR "'${NM}' failed on '${OBJECT}': ${NM_ERROR}")
  endif()

  string(REGEX MATCHALL "${REFCOUNT_REGEX}" REFCOUNT_SYMBOLS "${UNDEFINED_SYMBOLS}")
  list(JOIN REFCOUNT_SYMBOLS "\n" REFCOUNT_SYMBOLS_AS_TEXT)

  if(EXPECT STREQUAL "absent" AND REFCOUNT_SYMBOLS)
    message(
      FATAL_ERROR
        "'${OBJECT}' references reference counting functions, thus some moves were not optimized away:\n${REFCOUNT_SYMBOLS_AS_TEXT}"
    )
  elseif(EXPECT STREQUAL "present" AND NOT REFCOUNT_SYMBOLS)
    message(FATAL_ERROR "'${OBJECT}' was expected to reference reference counting functions, but does not.")
  endif()

  message(STATUS "'${OBJECT}': reference counting functions are ${EXPECT}, as expected.")
endforeach()
