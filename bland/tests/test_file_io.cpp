#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <bland/bland.hpp>
#include <bland_helpers/file_helpers.hpp>

#include <fmt/core.h>

#include <unistd.h>




TEST_CASE("file_io", "[ndarray][file]") {
    SECTION("write to file", "write tensor to a file") {
        char test_filename[] = "/tmp/bland_qa_XXXXXX"; // 'X's are placeholders
        int fd = mkstemp(test_filename);

        if (fd == -1) {
            throw std::runtime_error("Failed to create temporary file.");
        }

        // Generate an array and write it to disk
        auto x = bland::linspace(0.0f, 500.0f, 500);
        bland::write_to_file(x, test_filename);

        REQUIRE(get_fd_file_length(fd) == sizeof(float)*500);

        std::vector<float> as_read(500);
        read(fd, as_read.data(), sizeof(float)*500);

        REQUIRE_THAT(as_read,
                Catch::Matchers::Equals(std::vector<float>(x.data_ptr<float>(), x.data_ptr<float>()+500)));
    }

    SECTION("round trip to file", "send an array to a file then read it back") {
        char test_filename[] = "/tmp/bland_qa_XXXXXX"; // 'X's are placeholders
        int fd = mkstemp(test_filename);

        if (fd == -1) {
            throw std::runtime_error("Failed to create temporary file.");
        }

        // Generate an array and write it to disk
        auto x = bland::linspace(0.0f, 500.0f, 500);
        bland::write_to_file(x, test_filename);

        REQUIRE(get_fd_file_length(fd) == sizeof(float)*500);

        auto y = bland::read_from_file(test_filename, bland::ndarray::datatype::float32);

        REQUIRE_THAT(std::vector<float>(x.data_ptr<float>(), x.data_ptr<float>()+500),
                Catch::Matchers::Equals(std::vector<float>(y.data_ptr<float>(), y.data_ptr<float>()+500)));

    }
}
