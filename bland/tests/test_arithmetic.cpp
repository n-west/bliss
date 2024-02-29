
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "bland_matchers.hpp"

#include <fmt/core.h>
#include <fmt/format.h>
// #include <fmt/ranges.h>
#include <bland/bland.hpp>

TEST_CASE("cpu arithmetic", "[ops][arithmetic]") {
    SECTION("addition", "addition ops") {
        {
            auto test_array =
                    bland::arange(0.0f, 50.0f, 1.0f, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});

            auto x = test_array.reshape({5, 10});

            auto result = x + 10;

            REQUIRE(x.numel() == 50);

            REQUIRE_THAT(std::vector<float>(x.data_ptr<float>(), x.data_ptr<float>() + x.numel()),
                         BlandWithinAbs(std::vector<float>(test_array.data_ptr<float>(),
                                                           test_array.data_ptr<float>() + test_array.numel()),
                                        0.0001f));

            auto expected = bland::arange(10.0f, 60.0f, 1.0f, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});
            REQUIRE_THAT(std::vector<float>(result.data_ptr<float>(), result.data_ptr<float>() + result.numel()),
                         BlandWithinAbs(std::vector<float>(expected.data_ptr<float>(),
                                                           expected.data_ptr<float>() + expected.numel()),
                                        0.0001f));
        }
        { // second operand (rhs) is broadcasted
            auto test_array =
                    bland::arange(0.0f, 50.0f, 1.0f, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});

            auto a = test_array.reshape({5, 10});
            auto b = bland::arange(0.0f, 10.0f, 1.0f, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});

            auto result = a + b;

            REQUIRE(a.numel() == 50);
            REQUIRE(b.numel() == 10);

            auto expected = bland::arange(10.0f, 60.0f, 1.0f, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});
            REQUIRE_THAT(std::vector<float>(result.data_ptr<float>(), result.data_ptr<float>() + result.numel()),
                         BlandWithinAbs(
                                 std::vector<float>{
                                         0,  2,  4,  6,  8,  10, 12, 14, 16, 18,
                                         10, 12, 14, 16, 18, 20, 22, 24, 26, 28,
                                         20, 22, 24, 26, 28, 30, 32, 34, 36, 38,
                                         30, 32, 34, 36, 38, 40, 42, 44, 46, 48,
                                         40, 42, 44, 46, 48, 50, 52, 54, 56, 58,
                                 },
                                 0.0001f));
        }
    }
    SECTION("division", "division ops") {
        { // 2D divide by scalar
            auto test_array =
                    bland::arange(0.0f, 50.0f, 1.0f, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});

            auto x = test_array.reshape({5, 10});

            auto result = x / 10;

            REQUIRE(x.numel() == 50);

            REQUIRE_THAT(std::vector<float>(x.data_ptr<float>(), x.data_ptr<float>() + x.numel()),
                         BlandWithinAbs(std::vector<float>(test_array.data_ptr<float>(),
                                                           test_array.data_ptr<float>() + test_array.numel()),
                                        0.0001f));

            auto expected = bland::arange(0.0f, 5.0f, 0.1f, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});
            REQUIRE_THAT(std::vector<float>(result.data_ptr<float>(), result.data_ptr<float>() + result.numel()),
                         BlandWithinAbs(std::vector<float>(expected.data_ptr<float>(),
                                                           expected.data_ptr<float>() + expected.numel()),
                                        0.0001f));
        }
        { // 2D slice divide by scalar
            auto test_array =
                    bland::arange(0.0f, 50.0f, 1.0f, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});
            auto x = test_array.reshape({5, 10});

            auto x_slice = bland::slice(x, {1, 2, 3});

            auto result = x_slice / 10;
            auto expected =
                    bland::arange(2.0f, 52.0f, 10.f, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32}) / 10;
            REQUIRE_THAT(std::vector<float>(result.data_ptr<float>(), result.data_ptr<float>() + result.numel()),
                         BlandWithinAbs(std::vector<float>(expected.data_ptr<float>(),
                                                           expected.data_ptr<float>() + expected.numel()),
                                        0.0001f));
            expected = bland::arange(0.0f, 50.0f, 1.0f, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});
            REQUIRE_THAT(std::vector<float>(x.data_ptr<float>(), x.data_ptr<float>() + x.numel()),
                         BlandWithinAbs(std::vector<float>(expected.data_ptr<float>(),
                                                           expected.data_ptr<float>() + expected.numel()),
                                        0.0001f));
        }
        { // Do it again, but store the result in to the slice
            auto test_array =
                    bland::arange(0.0f, 50.0f, 1.0f, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});
            auto x = test_array.reshape({5, 10});

            auto x_slice = bland::slice(x, {1, 2, 3});

            x_slice = x_slice / 10;

            REQUIRE(x.numel() == 50);

            REQUIRE_THAT(std::vector<float>(x.data_ptr<float>(), x.data_ptr<float>() + x.numel()),
                         BlandWithinAbs(std::vector<float>(test_array.data_ptr<float>(),
                                                           test_array.data_ptr<float>() + test_array.numel()),
                                        0.0001f));

            REQUIRE_THAT(std::vector<float>(x.data_ptr<float>(), x.data_ptr<float>() + x.numel()),
                         BlandWithinAbs(
                                 std::vector<float>{
                                         0,  1,  .2, 3,  4,  5,   6,  7,  8,   9,  10, 11, 1.2, 13, 14, 15,  16,
                                         17, 18, 19, 20, 21, 2.2, 23, 24, 25,  26, 27, 28, 29,  30, 31, 3.2, 33,
                                         34, 35, 36, 37, 38, 39,  40, 41, 4.2, 43, 44, 45, 46,  47, 48, 49,
                                 },
                                 .0001f));
        }
        { // Slice both dims
            auto test_array =
                    bland::arange(0.0f, 50.0f, 1.0f, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});
            auto x = test_array.reshape({5, 10});

            auto x_slice = bland::slice(x, bland::slice_spec{0, 2, 3}, bland::slice_spec{1, 1, 4});

            x_slice = x_slice / 10;

            REQUIRE_THAT(std::vector<float>(x.data_ptr<float>(), x.data_ptr<float>() + x.numel()),
                         BlandWithinAbs(
                                 std::vector<float>{
                                         0,  1,  2,  3,  4,   5,   6,   7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                         17, 18, 19, 20, 2.1, 2.2, 2.3, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                                         34, 35, 36, 37, 38,  39,  40,  41, 42, 43, 44, 45, 46, 47, 48, 49,
                                 },
                                 .0001f));
        }
    }
}

#if BLAND_CUDA_CODE
#include <cuda_runtime.h>

TEST_CASE("cuda arithmetic", "[ops][arithmetic]") {
    SECTION("add array array", "[5, 10] array + [5, 10] array") {
            auto test_array =
                    bland::arange(0.0f, 50.0f, 1.0f, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});
            test_array = test_array.to("cuda:0");
            auto x = test_array.reshape({5, 10});

            auto result = x + x;
            result = result.to("cpu");
            cudaDeviceSynchronize();

            REQUIRE(x.numel() == 50);

            auto expected = bland::arange(0.0f, 100.0f, 2.0f, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});
            REQUIRE_THAT(std::vector<float>(result.data_ptr<float>(), result.data_ptr<float>() + result.numel()),
                         BlandWithinAbs(std::vector<float>(expected.data_ptr<float>(),
                                                           expected.data_ptr<float>() + expected.numel()),
                                        0.0001f));
    }
    SECTION("array-scalar mul-add", "[10000, 10000] array * 4 + 2") {
        auto test_array = bland::ndarray({10000, 10000}, 1.0f);

        test_array = test_array.to("cuda:0");

        auto result = test_array * 4 + 2;
        result = result.to("cpu");
        cudaDeviceSynchronize();

        REQUIRE(test_array.numel() == 10000*10000);
        auto expected = std::vector<float>(test_array.numel(), 6.0f);
        REQUIRE_THAT(std::vector<float>(result.data_ptr<float>(), result.data_ptr<float>() + result.numel()),
                        BlandWithinAbs(expected,
                                0.0001f));
    }
}
#endif

