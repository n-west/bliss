
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <fmt/format.h>

#include <bland/bland.hpp>

TEST_CASE("ndarray 1d stride", "[ndarray][stride]") {
    SECTION("stride1d", "stride a 1d array and use it to store results") {
        auto test_array = bland::arange(0, 20, 1, DLDataType{.code = DLDataTypeCode::kDLInt, .bits = 32});

        auto first_slice = bland::slice(test_array, bland::slice_spec{0, 0, 20, 2});

        REQUIRE(first_slice.ndim() == 1);
        REQUIRE(first_slice.numel() == 10);

        // The sliced array should have properties that match the slice
        REQUIRE_THAT(first_slice.shape(), Catch::Matchers::Equals(std::vector<int64_t>{10}));
        REQUIRE_THAT(first_slice.strides(), Catch::Matchers::Equals(std::vector<int64_t>{2}));
        REQUIRE_THAT(first_slice.offsets(), Catch::Matchers::Equals(std::vector<int64_t>{0}));

        // They should be the same buffer, strides are just views
        REQUIRE(first_slice.data_ptr<void>() == test_array.data_ptr<void>());

        // Can't do this properly until we have an ndarray equality test
        // REQUIRE_THAT(std::vector<int32_t>(first_slice.data_ptr<int32_t>(),
        //                                   first_slice.data_ptr<int32_t>() + first_slice.numel()),
        //              Catch::Matchers::Equals(std::vector<int32_t>{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20}));

        first_slice = bland::add(first_slice, first_slice);
        REQUIRE_THAT(std::vector<int32_t>(test_array.data_ptr<int32_t>(),
                                          test_array.data_ptr<int32_t>() + test_array.numel()),
                     Catch::Matchers::Equals(std::vector<int32_t>{0, 1, 4, 3, 8, 5, 12, 7, 16, 9, 20, 11, 24, 13, 28, 15, 32, 17, 36, 19}));

    }

    SECTION("stride1d", "stride a 1d array and use it to store scalar") {
        auto test_array = bland::arange(0, 20, 1, DLDataType{.code = DLDataTypeCode::kDLInt, .bits = 32});

        auto first_slice = bland::slice(test_array, bland::slice_spec{0, 0, 20, 2});

        REQUIRE(first_slice.ndim() == 1);
        REQUIRE(first_slice.numel() == 10);

        // The sliced array should have properties that match the slice
        REQUIRE_THAT(first_slice.shape(), Catch::Matchers::Equals(std::vector<int64_t>{10}));
        REQUIRE_THAT(first_slice.strides(), Catch::Matchers::Equals(std::vector<int64_t>{2}));
        REQUIRE_THAT(first_slice.offsets(), Catch::Matchers::Equals(std::vector<int64_t>{0}));

        // They should be the same buffer, strides are just views
        REQUIRE(first_slice.data_ptr<void>() == test_array.data_ptr<void>());

        // Can't do this properly until we have an ndarray equality test
        // REQUIRE_THAT(std::vector<int32_t>(first_slice.data_ptr<int32_t>(),
        //                                   first_slice.data_ptr<int32_t>() + first_slice.numel()),
        //              Catch::Matchers::Equals(std::vector<int32_t>{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20}));

        first_slice = bland::multiply(first_slice, 5);
        REQUIRE_THAT(std::vector<int32_t>(test_array.data_ptr<int32_t>(),
                                          test_array.data_ptr<int32_t>() + test_array.numel()),
                     Catch::Matchers::Equals(std::vector<int32_t>{0, 1, 10, 3, 20, 5, 30, 7, 40, 9, 50, 11, 60, 13, 70, 15, 80, 17, 90, 19}));

    }


    SECTION("stride2d", "stride a 2d array and make sure the shape is correct") {
        auto test_array = bland::ndarray({10, 10}, DLDataType{.code = DLDataTypeCode::kDLInt, .bits = 32});
        test_array = bland::fill(test_array, 1);

        auto slice = bland::slice(test_array, bland::slice_spec{0, 2, 4}, bland::slice_spec{1, 2, 4});

        // REQUIRE_THAT(slice.shape(), Catch::Matchers::Equals(std::vector<int64_t>{2,2});
        REQUIRE_THAT(slice.shape(), Catch::Matchers::Equals(std::vector<int64_t>{2,2}));
    }
}

#if BLAND_CUDA_CODE
#include <cuda_runtime.h>


TEST_CASE("cuda-slices", "[ndarray][stride]") {
    SECTION("sliced offset", "slice to make sure the operation starts at the correct offset") {
        auto test_array = bland::arange(0, 1000, 1, DLDataType{.code = DLDataTypeCode::kDLInt, .bits = 32});
        test_array = test_array.to("cuda:0");

        auto first_slice = bland::slice(test_array, bland::slice_spec{0, 5, 30, 1});

        first_slice = first_slice + 10;

        test_array = test_array.to("cpu");
        cudaDeviceSynchronize();

        REQUIRE_THAT(std::vector<int32_t>(test_array.data_ptr<int32_t>(),
                                            test_array.data_ptr<int32_t>() + 42),
                        Catch::Matchers::Equals(std::vector<int32_t>{0, 1, 2, 3, 4, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                        30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41}));
    }
    SECTION("broadcasting add", "broadcasting...") {
        auto test_array = bland::arange(0, 50, 1, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});
        auto test_array2 = test_array.to("cuda:0");

        auto test_2d = bland::ndarray({5, 50}, 40, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});
        bland::fill(bland::slice(test_2d, bland::slice_spec({0, 1, 2, 1})), 50);

        fmt::print("{}\n", test_2d.repr());
        test_2d = test_2d.to("cuda:0");

        cudaDeviceSynchronize();

        auto result = test_2d + test_array2;
        // auto result = test_array2 + test_2d;

        REQUIRE_THAT(result.shape(), Catch::Matchers::Equals(std::vector<int64_t>{5, 50}));
        // REQUIRE_THAT(slice.shape(), Catch::Matchers::Equals(std::vector<int64_t>{2,2}));

        REQUIRE(result.numel() == 250);

        result = result.to("cpu");
        cudaDeviceSynchronize();

        REQUIRE_THAT(std::vector<float>(result.data_ptr<float>(),
                                            result.data_ptr<float>() + 50),
            Catch::Matchers::Equals(std::vector<float>{40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
                                                        65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89}));

        REQUIRE_THAT(std::vector<float>(result.data_ptr<float>() + 50,
                                            result.data_ptr<float>() + 100),
            Catch::Matchers::Equals(std::vector<float>{50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                                                        80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99}));

    }
}

#endif