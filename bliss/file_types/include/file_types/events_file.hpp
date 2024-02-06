#pragma once

#if BLISS_USE_CAPNP

#include <core/hit.hpp>
#include <core/scan.hpp>
#include <core/cadence.hpp>
#include <core/event.hpp>

#include <string_view>
#include <vector>

namespace bliss {

/**
 * write all detected hits for all scans of each observation target in a cadence as cap'n proto messages to binary files matching
 * the file_path
 * the result will be one file per scan for each observation target with filenames matching the pattern
*/
void write_events_to_file(std::vector<event> events, std::string_view base_filename);

/**
 * read cap'n proto serialized scan from file as written by `write_scan_hits_to_file`
*/
std::vector<event> read_events_from_file(std::string_view file_path);


}

#endif // BLISS_USE_CAPNP
