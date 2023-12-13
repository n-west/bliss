
add_library(estimators
    noise_estimate.cpp
    spectral_kurtosis.cpp
)

target_include_directories(estimators
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(estimators
    PUBLIC
    bland
    bliss_core
    PRIVATE
    fmt::fmt-header-only
)


set_target_properties(estimators PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

add_library(pyestimators INTERFACE)
target_link_libraries(pyestimators INTERFACE estimators)
