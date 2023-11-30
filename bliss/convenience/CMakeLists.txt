
add_library(datatypedefs INTERFACE)

# target_link_libraries(datatypedefs INTERFACE Eigen3::Eigen)
target_include_directories(datatypedefs INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)