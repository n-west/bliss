
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "bland_matchers.hpp"

#include <fmt/core.h>
#include <bland/bland.hpp>

TEST_CASE("statistical", "[ndarray][ops][statistical]") {
    SECTION("mean", "test mean on variations of 1d, 2d, with and without axis arguments") {
        {
            auto test_array =
                    bland::arange(0.0f, 25.0f, 1.0f, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});

            auto result = bland::mean(test_array);

            REQUIRE_THAT(std::vector<float>(result.data_ptr<float>(), result.data_ptr<float>() + result.numel()),
                         Catch::Matchers::Equals(std::vector<float>{12}));
        }
        {
            auto test_array = bland::ndarray({16, 1}, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});
            bland::fill(test_array, 1);
            bland::fill(bland::slice(test_array, {0, 0, 17, 2}), 3);

            auto result = bland::mean(test_array);

            REQUIRE(result.ndim() == 1);
            REQUIRE_THAT(result.shape(), Catch::Matchers::Equals(std::vector<int64_t>{1}));

            REQUIRE_THAT(std::vector<float>(result.data_ptr<float>(), result.data_ptr<float>() + result.numel()),
                         Catch::Matchers::Equals(std::vector<float>{2}));
        }

        {
            auto test_array = bland::ndarray({5, 5}, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});
            bland::fill(test_array, 1);

            auto result = bland::mean(test_array);

            REQUIRE(result.ndim() == 1);
            REQUIRE_THAT(result.shape(), Catch::Matchers::Equals(std::vector<int64_t>{1}));

            REQUIRE_THAT(std::vector<float>(result.data_ptr<float>(), result.data_ptr<float>() + result.numel()),
                         Catch::Matchers::Approx(std::vector<float>{1}));
        }
        {
            auto test_array = bland::ndarray({5, 5}, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});
            bland::fill(test_array, 1);
            bland::fill(bland::slice(test_array, bland::slice_spec{0, 1, 3, 1}, bland::slice_spec{1, 2, 6, 1}), 3);
            /*
             * 1 1 1 1 1
             * 1 1 1 1 1
             * 1 3 3 1 1
             * 1 3 3 1 1
             * 1 3 3 1 1
             * */

            auto result = bland::mean(test_array, {1});

            REQUIRE(result.ndim() == 1);
            REQUIRE_THAT(result.shape(), Catch::Matchers::Equals(std::vector<int64_t>{5}));

            REQUIRE_THAT(std::vector<float>(result.data_ptr<float>(), result.data_ptr<float>() + result.numel()),
                         Catch::Matchers::Equals(std::vector<float>{1, 2.2, 2.2, 1, 1}));
        }
    }

    SECTION("standardized moments", "test kurtosis on variations of 1d, 2d, with and without axis arguments") {
        {
            auto test_array = bland::rand_normal(
                    {1000, 1000}, 0.0f, 1.0f, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});

            auto mean     = bland::mean(test_array);
            auto stddev   = bland::stddev(test_array);
            auto kurtosis = bland::standardized_moment(test_array, 4);
            REQUIRE_THAT(mean.data_ptr<float>()[0], Catch::Matchers::WithinAbs(0.0f, .1));
            REQUIRE_THAT(stddev.data_ptr<float>()[0], Catch::Matchers::WithinAbs(1.0f, .1));
            REQUIRE_THAT(kurtosis.data_ptr<float>()[0], Catch::Matchers::WithinAbs(3.0f, .1));

            // Changing the mean and variance, normalized moment should remain the same.
            test_array = test_array * 4 + 2;

            mean     = bland::mean(test_array);
            stddev   = bland::stddev(test_array);
            kurtosis = bland::standardized_moment(test_array, 4);
            REQUIRE_THAT(mean.data_ptr<float>()[0], Catch::Matchers::WithinAbs(2.0f, .1));
            REQUIRE_THAT(stddev.data_ptr<float>()[0], Catch::Matchers::WithinAbs(4.0f, .1));
            REQUIRE_THAT(kurtosis.data_ptr<float>()[0], Catch::Matchers::WithinAbs(3.0f, .11));
        }
        {
            auto test_array = bland::rand_normal(
                    {500000, 5}, 0.0f, 1.0f, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});

            bland::slice(test_array, {1, 1, 2}) = bland::rand_normal(
                    {500000, 1}, 2.0f, 4.0f, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32});
            bland::fill(bland::slice(test_array, {1, 2, 3}), 5);

            auto mean     = bland::mean(test_array, {0});
            auto stddev   = bland::stddev(test_array, {0});
            auto kurtosis = bland::standardized_moment(test_array, 4, {0});

            REQUIRE_THAT(std::vector<float>(mean.data_ptr<float>(), mean.data_ptr<float>() + mean.numel()),
                         BlandWithinAbs(std::vector<float>{0, 2, 5, 0, 0}, .1f));

            REQUIRE_THAT(std::vector<float>(stddev.data_ptr<float>(), stddev.data_ptr<float>() + stddev.numel()),
                         BlandWithinAbs(std::vector<float>{1, 4, 0, 1, 1}, .1f));

            REQUIRE_THAT(std::vector<float>(kurtosis.data_ptr<float>(), kurtosis.data_ptr<float>() + kurtosis.numel()),
                         BlandWithinAbs(std::vector<float>{3, 3, 0, 3, 3}, .1f));

            // Linear transform: mean and variance get scaled and translated, normalized moment should remain the same.
            test_array = test_array * 4 + 2;

            mean     = bland::mean(test_array, {0});
            stddev   = bland::stddev(test_array, {0});
            kurtosis = bland::standardized_moment(test_array, 4, {0});

            REQUIRE_THAT(std::vector<float>(mean.data_ptr<float>(), mean.data_ptr<float>() + mean.numel()),
                         BlandWithinAbs(std::vector<float>{2, 10, 22, 2, 2}, .1f));

            REQUIRE_THAT(std::vector<float>(stddev.data_ptr<float>(), stddev.data_ptr<float>() + stddev.numel()),
                         BlandWithinAbs(std::vector<float>{4, 16, 0, 4, 4}, .1f));

            REQUIRE_THAT(std::vector<float>(kurtosis.data_ptr<float>(), kurtosis.data_ptr<float>() + kurtosis.numel()),
                         BlandWithinAbs(std::vector<float>{3, 3, 0, 3, 3}, .1f));
        }
    }
}
