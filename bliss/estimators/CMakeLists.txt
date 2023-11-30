
add_library(estimators
    noise_estimate.cpp
)

target_include_directories(estimators
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(estimators
    PUBLIC
    bland
    datatypedefs
    PRIVATE
    fmt::fmt
)
