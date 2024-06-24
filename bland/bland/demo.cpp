
#include <bland/bland.hpp>

#include <fmt/core.h>

int main() {

    auto x = bland::ones({64}, bland::ndarray::datatype::float32);

    fmt::print("x = {}\n", x.repr());

    return 0;
}