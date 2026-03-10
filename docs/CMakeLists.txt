string(JSON DOXYGEN_VERSION GET "${DEPENDENCIES_JSON}" "github" "doxygen")

find_package(Doxygen ${DOXYGEN_VERSION} REQUIRED)

# Fetch tag files of other projects.
set(KokkosUtils_DOC_URL https://uliegecsm.github.io/kokkos-utils/)
file(DOWNLOAD ${KokkosUtils_DOC_URL}/kokkos-utils.tag ${CMAKE_CURRENT_BINARY_DIR}/kokkos-utils.tag
     STATUS fetch_tag_file
)
if(NOT fetch_tag_file EQUAL 0)
  message(
    FATAL_ERROR
      "An error occurred while fetching tag file for 'kokkos-utils' from ${KokkosUtils_DOC_URL} (${fetch_tag_file})"
  )
endif()
set(DOXYGEN_TAGFILES ${CMAKE_CURRENT_BINARY_DIR}/kokkos-utils.tag=${KokkosUtils_DOC_URL})

set(HOMEPAGE_MD ${CMAKE_SOURCE_DIR}/README.md)

set(DOXYGEN_CITE_BIB_FILES ${CMAKE_CURRENT_SOURCE_DIR}/references.bib)
set(DOXYGEN_ENABLE_PREPROCESSING YES)
set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_FILE_PATTERNS *.c *.cpp *.h *.hpp)
set(DOXYGEN_IMAGE_PATH ${CMAKE_SOURCE_DIR})
set(DOXYGEN_MACRO_EXPANSION YES)
set(DOXYGEN_REFERENCES_RELATION YES)
set(DOXYGEN_SOURCE_BROWSER YES)
set(DOXYGEN_TIMESTAMP YES)
set(DOXYGEN_USE_MDFILE_AS_MAINPAGE ${HOMEPAGE_MD})
set(DOXYGEN_WARN_AS_ERROR YES)
set(DOXYGEN_WARN_IF_UNDOCUMENTED NO)

get_target_property(GTest_gtest_INTERFACE_INCLUDE_DIRECTORIES GTest::gtest INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(Kokkos_kokkoscore_INTERFACE_INCLUDE_DIRECTORIES Kokkos::kokkoscore INTERFACE_INCLUDE_DIRECTORIES)
set(DOXYGEN_INCLUDE_PATH ${Kokkos_kokkoscore_INTERFACE_INCLUDE_DIRECTORIES}
                         ${GTest_gtest_INTERFACE_INCLUDE_DIRECTORIES} ${CMAKE_SOURCE_DIR}
)

doxygen_add_docs(docs ${HOMEPAGE_MD} ${CMAKE_SOURCE_DIR}/kokkos-execution ${CMAKE_SOURCE_DIR}/tests)
