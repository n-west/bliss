
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "bland_matchers.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <bland/bland.hpp>

TEST_CASE("ndarray 1d", "[ndarray]") {
    SECTION("create", "create 1d array and check various properties") {
        auto test_array = bland::ndarray({42}, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});
        REQUIRE(test_array.shape().size() == 1);
        REQUIRE(test_array.numel() == 42);
        REQUIRE(test_array.shape()[0] == 42);
        REQUIRE(test_array.strides()[0] == 1);
        REQUIRE(test_array.offsets()[0] == 0);

        REQUIRE_THAT(test_array.shape(), Catch::Matchers::Equals(std::vector<int64_t>{42}));
        REQUIRE_THAT(test_array.strides(), Catch::Matchers::Equals(std::vector<int64_t>{1}));
        REQUIRE_THAT(test_array.offsets(), Catch::Matchers::Equals(std::vector<int64_t>{0}));

        auto managed_tensor = test_array.get_managed_tensor();
        REQUIRE(managed_tensor->dl_tensor.data == test_array.data_ptr<void>());
        REQUIRE(managed_tensor->dl_tensor.ndim == 1);
        managed_tensor->deleter(managed_tensor);
        // TODO, think through how to test that we've safely shared with another framework even after our scope is gone
    }

    SECTION("reshape", "reshape 1d to 2d") {
        auto test_array =
                bland::arange(0.0f, 50.0f, 1.0f, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});

        auto x = test_array.reshape({5, 10});

        REQUIRE(x.numel() == 50);
        REQUIRE_THAT(x.shape(), Catch::Matchers::Equals(std::vector<int64_t>{5, 10}));
        REQUIRE_THAT(x.strides(), Catch::Matchers::Equals(std::vector<int64_t>{10, 1}));

        fmt::print("repr is {}\n", x.repr());
    }


    SECTION("reshape_large", "reshape larger array") {
        auto x =
                bland::linspace(0.0f, 512.0f, 512, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});

        auto y = x.reshape({16, 512/16});

        auto z = bland::add(y, y);

        fmt::print("repr is {}\n", z.repr());
        REQUIRE(z.numel() == 512);
        REQUIRE_THAT(x.shape(), Catch::Matchers::Equals(std::vector<int64_t>{16, 32}));
        REQUIRE_THAT(x.strides(), Catch::Matchers::Equals(std::vector<int64_t>{32, 1}));

    }
}
