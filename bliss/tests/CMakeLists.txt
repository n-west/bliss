
Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.3.0 # or a later release
)

FetchContent_MakeAvailable(Catch2)

add_executable(test_bliss
    test.cpp
    test_capnp_roundtrip.cpp
    )

target_link_libraries(test_bliss
  PRIVATE
  bliss_core
  bliss_serialization
  Catch2::Catch2
  fmt::fmt-header-only
  )

# # Helps auto discovery of test reporting to ctest
# list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
# include(CTest)
# include(Catch)
# catch_discover_tests(test_bland)
