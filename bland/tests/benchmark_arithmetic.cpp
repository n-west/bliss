#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <bland/bland.hpp>

TEST_CASE("benchmark add", "[ops][arithmetic]") {
    SECTION("addition", "addition") {
        {
            int64_t number_samples = 100000000;

            auto a = bland::rand_normal({number_samples}, 0.0f, 1.0f);
            auto b = bland::rand_normal({number_samples}, 0.0f, 1.0f);

            BENCHMARK("naive float addition") {
                auto c = bland::ndarray({number_samples}, DLDataType{.code = kDLFloat, .bits = 32});

                auto a_data = a.data_ptr<float>();
                auto b_data = b.data_ptr<float>();
                auto c_data = c.data_ptr<float>();
                for (size_t n = 0; n < number_samples; ++n) {
                    c_data[n] = a_data[n] + b_data[n];
                }
                return c;
            };

            BENCHMARK("bland float addition") {
                return a + b;
            };
        }
    }

    SECTION("add scalar", "add scalar") {
        {
            int64_t number_samples = 100000000;

            auto a = bland::rand_normal({number_samples}, 0.0f, 1.0f);
            float b = 42.0f;

            BENCHMARK("naive float addition") {
                auto c = bland::ndarray({number_samples}, DLDataType{.code = kDLFloat, .bits = 32});

                auto a_data = a.data_ptr<float>();
                auto c_data = c.data_ptr<float>();
                for (size_t n = 0; n < number_samples; ++n) {
                    c_data[n] = a_data[n] + b;
                }
                return c;
            };

            BENCHMARK("bland float addition") {
                return a + b;
            };
        }
    }
}
