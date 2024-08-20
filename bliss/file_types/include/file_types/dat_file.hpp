#pragma once

#include <core/hit.hpp>
#include <core/scan.hpp>
#include <core/cadence.hpp>

#include <string_view>
#include <vector>
#include <list>


namespace bliss {

// /**
//  * write hits as a .dat file similar to turboseti at the given path
// */
// template<template<typename> class Container>
// void write_hits_to_dat_file(Container<hit> hits, std::string_view file_path);

// /**
//  * read serialized hits from file as written by `write_hits_to_file` using given or assumed format
// */
// std::list<hit> read_hits_from_dat_file(std::string_view file_path);

/**
 * write hits as a .dat file similar to turboseti at the given path
*/
void write_scan_hits_to_dat_file(scan scan_with_hits, std::string_view file_path);

/**
 * read serialized hits from file as written by `write_hits_to_file` using given or assumed format
*/
scan read_scan_hits_from_dat_file(std::string_view file_path);

} // namespace bliss
