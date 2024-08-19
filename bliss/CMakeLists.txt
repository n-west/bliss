add_subdirectory(core)
add_subdirectory(estimators)
add_subdirectory(file_types)
add_subdirectory(flaggers)
add_subdirectory(preprocess)
add_subdirectory(rfi)
add_subdirectory(drift_search)

add_subdirectory(tests)
add_subdirectory(python)

#
# bliss_generate_channelizer_response
#
add_executable(bliss_generate_channelizer_response bliss_generate_channelizer_response.cpp)
target_link_libraries(bliss_generate_channelizer_response PUBLIC bliss_preprocess bland fmt::fmt-header-only)
install(TARGETS bliss_generate_channelizer_response RUNTIME COMPONENT bliss_executables DESTINATION bin)
# Installs executables to ship through python wheels
install(
  TARGETS bliss_generate_channelizer_response
  EXCLUDE_FROM_ALL
  RUNTIME DESTINATION ${PY_BUILD_CMAKE_PACKAGE_NAME}-${PY_BUILD_CMAKE_PACKAGE_VERSION}.data/scripts
          COMPONENT python_binaries
)

#
# bliss_find_hits
#
add_executable(bliss_find_hits bliss_find_hits.cpp)
target_link_libraries(
  bliss_find_hits
  PUBLIC bliss_core
         bliss_preprocess
         drift_search
         estimators
         bliss_serialization
         flaggers
         fmt::fmt-header-only
)

install(TARGETS bliss_find_hits RUNTIME COMPONENT bliss_executables DESTINATION bin)
# Installs executables to ship through python wheels
install(
  TARGETS bliss_find_hits
  EXCLUDE_FROM_ALL
  RUNTIME DESTINATION ${PY_BUILD_CMAKE_PACKAGE_NAME}-${PY_BUILD_CMAKE_PACKAGE_VERSION}.data/scripts
          COMPONENT python_binaries
)

#
# bliss_hits_to_dat
#
add_executable(bliss_hits_to_dat bliss_hits_to_dat.cpp)
target_link_libraries(bliss_hits_to_dat PUBLIC bliss_serialization fmt::fmt-header-only)
install(TARGETS bliss_hits_to_dat RUNTIME COMPONENT bliss_executables DESTINATION bin)
# Installs executables to ship through python wheels
install(
  TARGETS bliss_hits_to_dat
  EXCLUDE_FROM_ALL
  RUNTIME DESTINATION ${PY_BUILD_CMAKE_PACKAGE_NAME}-${PY_BUILD_CMAKE_PACKAGE_VERSION}.data/scripts
          COMPONENT python_binaries
)

#
# event_search
#
add_executable(event_search event_search.cpp)
target_link_libraries(event_search PUBLIC bliss_serialization drift_search fmt::fmt-header-only)
