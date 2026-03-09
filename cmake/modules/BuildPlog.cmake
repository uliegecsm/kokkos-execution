# Include Plog as suggested in https://github.com/SergiusTheBest/plog/issues/167#issuecomment-1431458232.
include(FetchContent)

FetchContent_Declare(plog SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/plog EXCLUDE_FROM_ALL)

set(PLOG_BUILD_SAMPLES OFF)
set(PLOG_INSTALL OFF)
set(PLOG_BUILD_TESTS OFF)

FetchContent_MakeAvailable(plog)

if(NOT TARGET plog::plog)
  message(FATAL_ERROR "Plog should define a plog::plog target.")
endif()
