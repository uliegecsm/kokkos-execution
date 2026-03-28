# Apply a patch once, and check it's been applied and don't re-apply it on subsequent calls.
function(apply_patch PATCH_PATH APPLY_DIR)

  find_package(Git REQUIRED)

  execute_process(
    COMMAND ${GIT_EXECUTABLE} apply --reverse --check ${PATCH_PATH}
    WORKING_DIRECTORY ${APPLY_DIR}
    RESULT_VARIABLE patch_already_applied
    OUTPUT_QUIET ERROR_QUIET
  )

  if(NOT patch_already_applied EQUAL 0)
    message(STATUS "Applying patch ${PATCH_PATH} to ${APPLY_DIR}.")
    execute_process(
      COMMAND ${GIT_EXECUTABLE} apply ${PATCH_PATH} WORKING_DIRECTORY ${APPLY_DIR} COMMAND_ERROR_IS_FATAL ANY
    )
  else()
    message(STATUS "Patch ${PATCH_PATH} already applied to ${APPLY_DIR}.")
  endif()
endfunction()
