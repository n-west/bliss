
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <bland/bland.hpp>

TEST_CASE("comparison", "[ndarray][boolean]") {
    SECTION("greater than scalar", "compare ndarray > scalar") {
        auto input = bland::arange(0, 20, 1, bland::ndarray::datatype::float32, bland::ndarray::dev::cpu);

        auto output = input > 10;

        REQUIRE_THAT(input.shape(), Catch::Matchers::Equals(output.shape()));
        REQUIRE(output.dtype().code == bland::ndarray::datatype::uint8.code);
        REQUIRE(output.dtype().bits == bland::ndarray::datatype::uint8.bits);
        
        REQUIRE_THAT(std::vector<uint8_t>(output.data_ptr<uint8_t>(), output.data_ptr<uint8_t>() + output.numel()),
                Catch::Matchers::Equals(std::vector<uint8_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                            1, 1, 1, 1, 1, 1, 1, 1, 1}));

        REQUIRE(bland::sum(output).scalarize<uint8_t>() == 9);
    }

    // SECTION("copy to existing", "copy from one tensor to another existing tensor of the same shape") {
    //     auto input = bland::ndarray({10, 10}, 42, bland::ndarray::datatype::uint8, bland::ndarray::dev::cpu);
    //     auto output = bland::ndarray({10, 10}, 100, bland::ndarray::datatype::uint8, bland::ndarray::dev::cpu);

    //     bland::copy(input, output);

    //     REQUIRE_THAT(input.shape(), Catch::Matchers::Equals(output.shape()));
    //     REQUIRE(input.dtype().code == output.dtype().code);
    //     REQUIRE(input.dtype().bits == output.dtype().bits);
        
    //     REQUIRE_THAT(std::vector<uint8_t>(output.data_ptr<uint8_t>(), output.data_ptr<uint8_t>() + output.numel()),
    //             Catch::Matchers::Equals(std::vector<uint8_t>{42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    //                                                         42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    //                                                         42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    //                                                         42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    //                                                         42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    //                                                         42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    //                                                         42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    //                                                         42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    //                                                         42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    //                                                         42, 42, 42, 42, 42, 42, 42, 42, 42, 42}));
    // }

}
