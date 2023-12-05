
add_library(flaggers
    kurtosis.cpp
)

target_include_directories(flaggers
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(flaggers
    PUBLIC
    bland
    datatypedefs
    PRIVATE
    fmt::fmt
)
