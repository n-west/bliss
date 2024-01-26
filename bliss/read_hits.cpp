
#include <file_types/hits_file.hpp>

#include "fmt/core.h"
#include <fmt/ranges.h>

#include <cstdint>
#include <string>
#include <vector>
#include <iostream>

int main(int argc, char **argv) {

    std::string hits_path{"serialized_hits.cp"};
    if (argc == 2) {
        hits_path = argv[1];
    }

    auto hits = bliss::read_hits_from_file(hits_path);

    std::cout << "There are " << hits.size() << " hits in this file\n";

}
