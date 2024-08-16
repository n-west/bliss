#pragma once

#if BLISS_USE_CAPNP


#include <string_view>
#include <vector>

namespace bliss {

struct event;

/**
 * write all detected hits for all scans of each observation target in a cadence as cap'n proto messages to binary files matching
 * the file_path
 * the result will be one file per scan for each observation target with filenames matching the pattern
*/
void write_events_to_file(std::vector<event> events, std::string_view base_filename);

/**
 * read cap'n proto serialized scan from file as written by `write_coarse_channel_hits_to_capnp_file`
*/
std::vector<event> read_events_from_file(std::string_view file_path);


}

#endif // BLISS_USE_CAPNP
