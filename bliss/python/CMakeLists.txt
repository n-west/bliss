nanobind_add_module(pybliss pybliss.cpp)
target_link_libraries(
  pybliss
  PUBLIC pybliss_core
         pybland_header
         pybliss_preprocess
         pydrift_search
         pyestimators
         pyfile_types
         pyflaggers
  PRIVATE fmt::fmt-header-only
)

if(WITH_CUDA)
  target_link_libraries(pybliss PRIVATE CUDA::cudart_static)
endif()

file(GLOB_RECURSE PYTHON_FILES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/blissdedrift/*.py)

foreach(PY_FILE ${PYTHON_FILES})
  file(RELATIVE_PATH REL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/blissdedrift ${PY_FILE})
  get_filename_component(DIR_PATH ${REL_PATH} DIRECTORY)
  set(OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/blissdedrift/${REL_PATH}")

  add_custom_command(
    OUTPUT ${OUTPUT_PATH}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/blissdedrift/${DIR_PATH}"
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${PY_FILE} ${OUTPUT_PATH}
    DEPENDS ${PY_FILE}
    COMMENT "Updating ${REL_PATH}"
  )

  list(APPEND COPIED_PYTHON_FILES ${OUTPUT_PATH})
endforeach()

add_custom_target(UpdatePythonFiles ALL DEPENDS ${COPIED_PYTHON_FILES})

add_custom_target(bliss-module ALL)
add_dependencies(bliss-module UpdatePythonFiles pybliss pybland)

add_custom_command(
  TARGET bliss-module
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:pybliss>
          "${CMAKE_CURRENT_BINARY_DIR}/blissdedrift/$<TARGET_FILE_NAME:pybliss>"
  COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:pybland>
          "${CMAKE_CURRENT_BINARY_DIR}/blissdedrift/$<TARGET_FILE_NAME:pybland>"
)

# Check if PY_BUILD_CMAKE_MODULE_NAME is defined
if(DEFINED PY_BUILD_CMAKE_MODULE_NAME)
  # Use the defined value
  set(PYBLISS_MODULE_INSTALL_DIR ${PY_BUILD_CMAKE_MODULE_NAME})
else()
  # Set a default value (e.g., install in site-packages)
  set(PYBLISS_MODULE_INSTALL_DIR "lib/python${PYTHON_VERSION}/site-packages")
endif()

install(
  TARGETS pybliss
  COMPONENT python_modules
  LIBRARY DESTINATION ${PYBLISS_MODULE_INSTALL_DIR}
)
