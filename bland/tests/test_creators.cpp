
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <bland/bland.hpp>
#include <fmt/core.h>

TEST_CASE("arange", "[ndarray][creator]") {
    SECTION("0:20", "create arange(0, 20)") {
        auto test_array = bland::arange(0, 20, 1, DLDataType{.code = DLDataTypeCode::kDLInt, .bits = 32});

        REQUIRE_THAT(std::vector<int32_t>(test_array.data_ptr<int32_t>(),
                                          test_array.data_ptr<int32_t>() + test_array.numel()),
                     Catch::Matchers::Equals(std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}));
    }
}

TEST_CASE("linspace", "[ndarray][creator]") {
    SECTION("0:20:2", "create 10 steps between 0 and 20") {
        auto test_array = bland::linspace(0, 20, 10, DLDataType{.code = DLDataTypeCode::kDLInt, .bits = 32});

        REQUIRE_THAT(std::vector<int32_t>(test_array.data_ptr<int32_t>(),
                                          test_array.data_ptr<int32_t>() + test_array.numel()),
                     Catch::Matchers::Equals(std::vector<int32_t>{0, 2, 4, 6, 8, 10, 12, 14, 16, 18}));
    }
}

TEST_CASE("zeros", "[ndarray][creator]") {
    SECTION("20 0s", "generate a new tensor initialized with 0s") {
        auto test_array = bland::linspace(0, 20, 20, DLDataType{.code = DLDataTypeCode::kDLInt, .bits = 32});
        bland::fill(test_array, (int32_t)0);
        REQUIRE_THAT(std::vector<int32_t>(test_array.data_ptr<int32_t>(),
                                          test_array.data_ptr<int32_t>() + test_array.numel()),
                     Catch::Matchers::Equals(std::vector<int32_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
    }

    SECTION("15 1s", "generate a new tensor initialized with 1s") {
        auto test_array = bland::linspace(0, 15, 15, DLDataType{.code = DLDataTypeCode::kDLInt, .bits = 32});
        bland::fill(test_array, (int32_t)1);
        REQUIRE_THAT(std::vector<int32_t>(test_array.data_ptr<int32_t>(),
                                          test_array.data_ptr<int32_t>() + test_array.numel()),
                     Catch::Matchers::Equals(std::vector<int32_t>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ,1, 1, 1, 1}));
    }

    SECTION("slice fill", "fill in slice") {
        auto test_array = bland::linspace(0, 15, 15, DLDataType{.code = DLDataTypeCode::kDLInt, .bits = 32});
        bland::fill(test_array, (int32_t)1);
        bland::fill(bland::slice(test_array, {0, 2, 4, 1}), 0);
        REQUIRE_THAT(std::vector<int32_t>(test_array.data_ptr<int32_t>(),
                                          test_array.data_ptr<int32_t>() + test_array.numel()),
                     Catch::Matchers::Equals(std::vector<int32_t>{1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1 ,1, 1, 1, 1}));
    }
}


TEST_CASE("random uniform", "[creator][random]") {
    SECTION("1000000 float uniform", "generate a new tensor with uniform random values between 0 and 42") {
        auto test_array = bland::rand_uniform({1000000}, 0, 42, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});

        auto mean = bland::mean(test_array);
        
        REQUIRE_THAT(mean.data_ptr<float>()[0], Catch::Matchers::WithinAbs(21, .2));
    }

    SECTION("1000000 int uniform", "generate a new tensor with uniform random values between 0 and 42") {
        auto test_array = bland::rand_uniform({1000000}, 0, 42, DLDataType{.code = DLDataTypeCode::kDLInt, .bits = 32});

        auto mean = bland::mean(test_array);
        
        // We're random, so this does occasionally wiggle
        REQUIRE(std::abs(mean.data_ptr<int32_t>()[0]-21) <= 1);
    }

}

TEST_CASE("random normal", "[creator][random]") {
    SECTION("1000000 float normal", "generate a new tensor with uniform normal values u=0, σ=1") {
        auto test_array = bland::rand_normal({1000000}, 0.0f, 1.0f, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});

        auto mean = bland::mean(test_array);
        auto stddev = bland::stddev(test_array);
        
        REQUIRE_THAT(mean.data_ptr<float>()[0], Catch::Matchers::WithinAbs(0, .1));
        REQUIRE_THAT(stddev.data_ptr<float>()[0], Catch::Matchers::WithinAbs(1, .1));
    }

    SECTION("1000000 double normal with mean and stddev", "generate a new tensor with uniform normal values u=4, σ=2") {
        auto test_array = bland::rand_normal({1000000}, 4.0f, 2.0f, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});

        auto mean = bland::mean(test_array);
        auto stddev = bland::stddev(test_array);
        
        REQUIRE_THAT(mean.data_ptr<float>()[0], Catch::Matchers::WithinAbs(4, .1));
        REQUIRE_THAT(stddev.data_ptr<float>()[0], Catch::Matchers::WithinAbs(2, .1));
    }

}
