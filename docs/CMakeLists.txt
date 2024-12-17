
find_package(Sphinx REQUIRED)


set(DOXYGEN_INPUT_DIR ${CMAKE_SOURCE_DIR})
set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs/doxygen)
set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html)

# Set input and output files
set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

# Request to configure the file
configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)
file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})
message("Doxygen build started")

add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
    #DEPENDS bliss # Actually want just the public headers!
                    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
                    MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
                    COMMENT "Generating docs with Doxygen (new)"
                    )

add_custom_target(doc ALL DEPENDS ${DOXYGEN_INDEX_FILE})
# Note the option ALL which allows to build the docs together with the application
# add_custom_target(
#   doc
#   COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
#   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#   COMMENT "Generating docs with Doxygen"
#   VERBATIM
# )
# set_target_properties(doc PROPERTIES EXCLUDE_FROM_ALL TRUE)


set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/docs/sphinx)

add_custom_target(Sphinx ALL
                    DEPENDS ${DOXYGEN_INDEX_FILE}
                    COMMAND
                    ${SPHINX_BUILD_EXECUTABLE} -b html
                    -Dbreathe_projects.bliss=${DOXYGEN_OUTPUT_DIR}/xml
                    ${SPHINX_SOURCE} ${SPHINX_BUILD}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                    COMMENT "Generating docs with sphinx")
