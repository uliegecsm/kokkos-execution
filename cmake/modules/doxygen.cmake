# Fetch a Doxygen tag file from a remote documentation site.
#
# Usage: fetch_doxygen_tag(<project> <doc_url>)
#
# Sets DOXYGEN_TAGFILES in the parent scope, appending to any existing value.
function(fetch_doxygen_tag PROJECT URL)
  set(tag_file "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT}.tag")

  file(DOWNLOAD "${URL}/${PROJECT}.tag" "${tag_file}" STATUS status)
  list(GET status 0 error_code)
  list(GET status 1 error_message)

  if(NOT error_code EQUAL 0)
    message(FATAL_ERROR "Failed to fetch tag file for '${PROJECT}' from ${URL}:\n\t${error_message}")
  endif()

  if(DEFINED DOXYGEN_TAGFILES)
    list(APPEND DOXYGEN_TAGFILES "${tag_file}=${URL}")
  else()
    set(DOXYGEN_TAGFILES "${tag_file}=${URL}")
  endif()
  set(DOXYGEN_TAGFILES "${DOXYGEN_TAGFILES}" PARENT_SCOPE)
endfunction()
