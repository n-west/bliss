#pragma once

#include <core/hit.hpp>
#include <core/scan.hpp>
#include <core/cadence.hpp>

#include <string_view>
#include <vector>
#include <list>

namespace bliss {

template<template<typename> class Container>
void write_hits_to_file(Container<hit> hits, std::string_view file_path, std::string format="");

std::list<hit> read_hits_from_file(std::string_view file_path, std::string format="");

void write_scan_hits_to_file(scan scan_with_hits, std::string_view file_path, std::string format="");

scan read_scan_hits_from_file(std::string_view file_path, std::string format="");

} // namespace bliss
