
#include <bland/bland.hpp>

#include <iostream>

int main () {


    auto x = bland::linspace(0, 20, 20);
    std::cout << x.repr() << std::endl;


    auto y = bland::slice(x, {0, 1, 20, 4});
    std::cout << y.repr() << std::endl;


    // y = bland::slice(x, 0, 0, 19, 4);
    // std::cout << y.repr() << std::endl;

    y = bland::add(y, y);
    std::cout << "y is " << y.repr() << std::endl;
    std::cout << "x is " << x.repr() << std::endl;
    std::cout << y.repr() << std::endl;

    y = bland::slice(y, {0, 2, 4, 1});
    std::cout << y.repr() << std::endl;

}
