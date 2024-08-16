#pragma once

#if BLISS_USE_CAPNP

#include <core/hit.hpp>
#include <core/scan.hpp>
#include <core/cadence.hpp>

#include <string_view>
#include <vector>
#include <list>

namespace bliss {

/**
 * write hits as independently serialized cap'n proto messages packed in to a binary file at the given path
*/
template<template<typename> class Container>
void write_hits_to_capnp_file(Container<hit> hits, std::string_view file_path);

/**
 * read cap'n proto serialized hits from file as written by `write_hits_to_capnp_file`
*/
std::list<hit> read_hits_from_capnp_file(std::string_view file_path);

/**
 * write scan metadata and associated hits as cap'n proto messages to binary file at the given path
*/
void write_coarse_channel_hits_to_capnp_file(coarse_channel scan_with_hits, std::string_view file_path);

/**
 * read cap'n proto serialized scan from file as written by `write_coarse_channel_hits_to_capnp_file`
*/
coarse_channel read_coarse_channel_hits_from_capnp_file(std::string_view file_path);

/**
 * write scan metadata and associated hits as cap'n proto messages to binary file at the given path
*/
void write_scan_hits_to_capnp_file(scan scan_with_hits, std::string_view file_path);

/**
 * read cap'n proto serialized scan from file as written by `write_coarse_channel_hits_to_capnp_file`
*/
scan read_scan_hits_from_capnp_file(std::string_view file_path);

/**
 * write an observation target's scan md and associated hits as cap'n proto messages to binary files matching
 * the file_path
 * the result will be one file per scan of the observation target
*/
void write_observation_target_hits_to_capnp_files(observation_target scan_with_hits, std::string_view file_path);

/**
 * read cap'n proto serialized scan from file as written by `write_coarse_channel_hits_to_capnp_file`
*/
observation_target read_observation_target_hits_from_capnp_files(std::string_view file_path);

/**
 * write all detected hits for all scans of each observation target in a cadence as cap'n proto messages to binary files matching
 * the file_path
 * the result will be one file per scan for each observation target with filenames matching the pattern
*/
void write_cadence_hits_to_capnp_files(cadence cadence_with_hits, std::string_view file_path);

/**
 * read cap'n proto serialized scan from file as written by `write_coarse_channel_hits_to_capnp_file`
*/
cadence read_cadence_hits_from_capnp_files(std::string_view file_path);


} // namespace bliss

#endif // BLISS_USE_CAPNP
