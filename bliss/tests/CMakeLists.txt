
Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.0.1 # or a later release
)

FetchContent_MakeAvailable(Catch2)

add_executable(test_bliss
    test.cpp
    )

target_link_libraries(test_bliss
  PRIVATE
  bland
  Catch2::Catch2
  fmt::fmt
  )

# # Helps auto discovery of test reporting to ctest
# list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
# include(CTest)
# include(Catch)
# catch_discover_tests(test_bland)
