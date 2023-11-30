

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <bland/bland.hpp>

TEST_CASE("ops", "[ndarray][ops]") {
    SECTION("sum", "test sum on variations of 1d, 2d, with and without axis arguments") {
    {
        auto test_array = bland::arange(0.0f, 25.0f, 1.0f, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});

        auto result = bland::mean(test_array);

        REQUIRE_THAT(std::vector<float>(result.data_ptr<float>(),
                                          result.data_ptr<float>() + result.numel()),
                     Catch::Matchers::Equals(std::vector<float>{12}));
    }
    {
        auto test_array = bland::ndarray({15,1}, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});
        bland::fill(test_array, 1);

        auto result = bland::sum(test_array);

        REQUIRE_THAT(std::vector<float>(result.data_ptr<float>(),
                                          result.data_ptr<float>() + result.numel()),
                     Catch::Matchers::Equals(std::vector<float>{15}));
    }

    {
        auto test_array = bland::ndarray({5,5}, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});
        bland::fill(test_array, 1);

        auto result = bland::sum(test_array);

        REQUIRE_THAT(std::vector<float>(result.data_ptr<float>(),
                                          result.data_ptr<float>() + result.numel()),
                     Catch::Matchers::Equals(std::vector<float>{25}));
    }
    {
        auto test_array = bland::ndarray({5,5}, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});
        bland::fill(test_array, 1);
        bland::fill(bland::slice(test_array, bland::slice_spec{0, 1, 3, 1}, bland::slice_spec{1, 2, 6, 1}), 3);
        /*
         * 1 1 1 1 1
         * 1 1 1 1 1
         * 1 3 3 1 1
         * 1 3 3 1 1
         * 1 3 3 1 1
         * =========
         * 5 11 11 5 5
         * */

        auto result = bland::sum(test_array, {1});

        REQUIRE_THAT(std::vector<float>(result.data_ptr<float>(),
                                          result.data_ptr<float>() + result.numel()),
                     Catch::Matchers::Equals(std::vector<float>{5, 11, 11, 5, 5}));
    }

    {
        auto test_array = bland::ndarray({5,5}, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});
        bland::fill(test_array, 1);

        auto result = bland::sum(test_array, {0,1});

        REQUIRE_THAT(std::vector<float>(result.data_ptr<float>(),
                                          result.data_ptr<float>() + result.numel()),
                     Catch::Matchers::Equals(std::vector<float>{25,}));
    }

    {
        auto test_array = bland::ndarray({5,5}, DLDataType{.code = DLDataTypeCode::kDLInt, .bits = 64});
        bland::fill(test_array, 1);

        auto result = bland::sum(test_array, {0,1});

        REQUIRE_THAT(std::vector<int64_t>(result.data_ptr<int64_t>(),
                                          result.data_ptr<int64_t>() + result.numel()),
                     Catch::Matchers::Equals(std::vector<int64_t>{25,}));
    }

    }
}
