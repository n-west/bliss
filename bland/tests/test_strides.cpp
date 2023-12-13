
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

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
