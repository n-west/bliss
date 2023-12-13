
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <bland/bland.hpp>

TEST_CASE("copy", "[ndarray][copy]") {
    SECTION("copy to new", "copy from one tensor to a new tensor") {
        auto input = bland::ndarray({10, 10}, 42, bland::ndarray::datatype::uint8, bland::ndarray::dev::cpu);

        auto output = bland::copy(input);

        REQUIRE_THAT(input.shape(), Catch::Matchers::Equals(output.shape()));
        REQUIRE(input.dtype().code == output.dtype().code);
        REQUIRE(input.dtype().bits == output.dtype().bits);
        
        REQUIRE_THAT(std::vector<uint8_t>(output.data_ptr<uint8_t>(), output.data_ptr<uint8_t>() + output.numel()),
                Catch::Matchers::Equals(std::vector<uint8_t>{42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                                                            42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                                                            42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                                                            42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                                                            42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                                                            42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                                                            42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                                                            42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                                                            42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                                                            42, 42, 42, 42, 42, 42, 42, 42, 42, 42}));
    }

    SECTION("copy to existing", "copy from one tensor to another existing tensor of the same shape") {
        auto input = bland::ndarray({10, 10}, 42, bland::ndarray::datatype::uint8, bland::ndarray::dev::cpu);
        auto output = bland::ndarray({10, 10}, 100, bland::ndarray::datatype::uint8, bland::ndarray::dev::cpu);

        bland::copy(input, output);

        REQUIRE_THAT(input.shape(), Catch::Matchers::Equals(output.shape()));
        REQUIRE(input.dtype().code == output.dtype().code);
        REQUIRE(input.dtype().bits == output.dtype().bits);
        
        REQUIRE_THAT(std::vector<uint8_t>(output.data_ptr<uint8_t>(), output.data_ptr<uint8_t>() + output.numel()),
                Catch::Matchers::Equals(std::vector<uint8_t>{42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                                                            42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                                                            42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                                                            42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                                                            42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                                                            42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                                                            42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                                                            42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                                                            42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                                                            42, 42, 42, 42, 42, 42, 42, 42, 42, 42}));
    }

    // SECTION("broadcast copy to existing", "copy from one tensor to another existing tensor using a broadcast") {
    //     auto input = bland::ndarray({1, 10}, 42, bland::ndarray::datatype::uint8, bland::ndarray::dev::cpu);
    //     auto output = bland::ndarray({10, 10}, 100, bland::ndarray::datatype::uint8, bland::ndarray::dev::cpu);

    //     bland::copy(input, output);

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

    SECTION("broadcast copy to existing", "copy from one tensor to another existing tensor using a broadcast") {
        auto input = bland::arange(0, 10, 1, bland::ndarray::datatype::uint8, bland::ndarray::dev::cpu);
        input = input.unsqueeze(0);
        auto output = bland::ndarray({10, 10}, 100, bland::ndarray::datatype::uint8, bland::ndarray::dev::cpu);

        bland::copy(input, output);

        REQUIRE(input.dtype().code == output.dtype().code);
        REQUIRE(input.dtype().bits == output.dtype().bits);
        
        REQUIRE_THAT(std::vector<uint8_t>(output.data_ptr<uint8_t>(), output.data_ptr<uint8_t>() + output.numel()),
                Catch::Matchers::Equals(std::vector<uint8_t>{   0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
    }
}
