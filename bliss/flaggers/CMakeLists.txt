
add_library(flaggers
    magnitude.cpp
    spectral_kurtosis.cpp
)

target_include_directories(flaggers
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(flaggers
    PUBLIC
    bland
    datatypedefs
    file_types
    PRIVATE
    estimators
    fmt::fmt-header-only
)


set_target_properties(flaggers PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

add_library(pyflaggers INTERFACE)
# target_include_directories()
target_link_libraries(pyflaggers INTERFACE flaggers)
