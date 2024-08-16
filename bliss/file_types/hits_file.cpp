
#include "file_types/hits_file.hpp"

#include "file_types/dat_file.hpp"
#include "file_types/cpnp_files.hpp"

#include <fmt/format.h>

#include <string_view>
#include <string>

using namespace bliss;

// In c++20 this comes with std::string
static bool ends_with(std::string_view str, std::string_view suffix) {
    return str.size() >= suffix.size() && str.compare(str.size()-suffix.size(), suffix.size(), suffix) == 0;
}

template<template<typename> class Container>
void bliss::write_hits_to_file(Container<hit> hits, std::string_view file_path, std::string format) {
    if (format.empty()) {
        if (ends_with(file_path, ".dat")) {
            format = "dat";
        } else if (ends_with(file_path, ".capnp") || ends_with(file_path, ".cp")) {
            format = "capnp";
        }
    }

    // TODO: decide and canonicalize a good default after talking with more users
    if (format == "dat") {
        write_hits_to_dat_file(hits, file_path);
    } else if (format == "capnp") {
        write_hits_to_capnp_file(hits, file_path);
    } else {
        fmt::print("INFO: No format specified while writing hits to disk. Defaulting to capnp serialization\n");
        write_hits_to_capnp_file(hits, file_path);
    }
}

std::list<hit> bliss::read_hits_from_file(std::string_view file_path, std::string format) {
    if (format.empty() || format == "capnp") {
        return read_hits_from_capnp_file(file_path);
    } else if (format == "dat" || format == "turboseti") {
        return read_hits_from_dat_file(file_path);
    }
}

void bliss::write_scan_hits_to_file(scan scan_with_hits, std::string_view file_path, std::string format) {
    if (format.empty()) {
        if (ends_with(file_path, ".dat")) {
            format = "dat";
        } else if (ends_with(file_path, ".capnp") || ends_with(file_path, ".cp")) {
            format = "capnp";
        }
    }

    // TODO: decide and canonicalize a good default after talking with more users
    if (format == "dat") {
        write_scan_hits_to_dat_file(scan_with_hits, file_path);
    } else if (format == "capnp") {
        write_scan_hits_to_capnp_file(scan_with_hits, file_path);
    } else {
        fmt::print("INFO: No format specified while writing hits to disk. Defaulting to capnp serialization\n");
        write_scan_hits_to_capnp_file(scan_with_hits, file_path);
    }
}

scan bliss::read_scan_hits_from_file(std::string_view file_path, std::string format) {

}

