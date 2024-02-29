#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <bland/bland.hpp>

#if BLAND_CUDA_CODE
#include <cuda_runtime.h>
#endif

TEST_CASE("benchmark device mean", "[benchmark]") {
    SECTION("mean", "1-dim mean") {
        {

            float number_samples = 1000000.0f;
            auto test_array =
                    bland::arange(0.0f, number_samples, 1.0f, DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32}, bland::ndarray::dev::cpu);

            BENCHMARK("mean on cpu") {
                return bland::mean(test_array);
            };

#if BLAND_CUDA_CODE
            test_array = bland::to(test_array, "cuda:0");
            BENCHMARK("mean on cuda:0") {
                auto r = bland::mean(test_array);
                cudaDeviceSynchronize();
                return r;
            };
#endif
        }
    }

}
