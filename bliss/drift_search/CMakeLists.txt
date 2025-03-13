
add_library(drift_search_kernels
    # dedoppler integration
    kernels/drift_integration_bland.cpp
    kernels/drift_integration_cpu.cpp
    # hit search
    kernels/local_maxima_cpu.cpp
    kernels/connected_components_cpu.cpp
)

if (WITH_CUDA)
    target_sources(drift_search_kernels
        PRIVATE
        # dedoppler integration
        kernels/drift_integration_cuda.cu
        # hit searches
        kernels/connected_components_cuda.cu
        kernels/connected_components_driftblock_cuda.cu
        kernels/local_maxima_cuda.cu
        )
endif()

target_link_libraries(drift_search_kernels
    PUBLIC
    bland
    bliss_core
    PRIVATE
    fmt::fmt-header-only
    drift_search # this makes it circular, primarily because of bliss::component
)

if (WITH_CUDA)
    target_link_libraries(drift_search_kernels
        PRIVATE
        CUDA::cudart_static
    )
endif()

set_target_properties(drift_search_kernels PROPERTIES POSITION_INDEPENDENT_CODE TRUE)


add_library(drift_search
    integrate_drifts.cpp
    protohit_search.cpp
    hit_search.cpp
    filter_hits.cpp
    event_search.cpp
    # hit search kernels
    connected_components.cpp
    local_maxima.cpp
)

target_include_directories(drift_search
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(drift_search
    PUBLIC
    bland
    bliss_core
    PRIVATE
    fmt::fmt-header-only
    flaggers
    drift_search_kernels
)

set_target_properties(drift_search PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

add_library(pydrift_search INTERFACE)
target_link_libraries(pydrift_search INTERFACE drift_search)

