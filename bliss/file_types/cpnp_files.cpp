
#if BLISS_USE_CAPNP

#include "file_types/cpnp_files.hpp"
#include "detail/raii_file_helpers.hpp"
#include "detail/cpnp_hit_builder.hpp"

#include "hit.capnp.h"

#include <stdexcept>

#include <capnp/c++.capnp.h>
#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/exception.h>

#include <fmt/core.h>
#include <fmt/format.h>

using namespace bliss;


template<template<typename> class Container>
void bliss::write_hits_to_capnp_file(Container<hit> hits, std::string_view file_path) {

    auto out_file = detail::raii_file_for_write(file_path);

    for (auto &this_hit : hits) {
        capnp::MallocMessageBuilder message;
        auto                        hit_builder    = message.initRoot<Hit>();
        auto                        signal_builder = hit_builder.initSignal();
        detail::bliss_hit_to_capnp_signal_message(signal_builder, this_hit);

        capnp::writeMessageToFd(out_file._fd, message);
    }
}
template void bliss::write_hits_to_capnp_file<std::vector>(std::vector<hit> hits, std::string_view file_path);
template void bliss::write_hits_to_capnp_file<std::list>(std::list<hit> hits, std::string_view file_path);


std::list<hit> bliss::read_hits_from_capnp_file(std::string_view file_path) {
    auto in_file = detail::raii_file_for_read(file_path);

    std::list<hit> hits;

    while (true) {
        try {
            capnp::StreamFdMessageReader message(in_file._fd);
            auto                         hit_reader    = message.getRoot<Hit>();
            auto                         signal_reader = hit_reader.getSignal();

            auto this_hit = detail::capnp_signal_message_to_bliss_hit(signal_reader);

            hits.push_back(this_hit);
        } catch (kj::Exception &e) {
            // We've reached the end of the file.
            break;
        }
    }

    return hits;
}

void bliss::write_coarse_channel_hits_to_capnp_file(coarse_channel cc, std::string_view file_path) {
    auto out_file = detail::raii_file_for_write(file_path);

    capnp::MallocMessageBuilder message;
    auto cc_message = message.initRoot<CoarseChannel>();

    detail::bliss_coarse_channel_to_capnp_coarse_channel_message(cc_message, cc);

    capnp::writeMessageToFd(out_file._fd, message);
}

coarse_channel bliss::read_coarse_channel_hits_from_capnp_file(std::string_view file_path) {

    auto in_file = detail::raii_file_for_read(file_path);

    capnp::StreamFdMessageReader message(in_file._fd);

    auto coarse_channel_reader = message.getRoot<CoarseChannel>();
    auto deserialized_cc = detail::capnp_coarse_channel_message_to_bliss_coarse_channel(coarse_channel_reader);

    return deserialized_cc; // return a copy, unique ptr will dealloc after copy
}

void bliss::write_scan_hits_to_capnp_file(scan scan_with_hits, std::string_view file_path) {
    auto out_file = detail::raii_file_for_write(file_path);

    capnp::MallocMessageBuilder message;
    auto scan_message = message.initRoot<Scan>();

    detail::bliss_scan_to_capnp_coarse_scan(scan_message, scan_with_hits);

    capnp::writeMessageToFd(out_file._fd, message);
}

scan bliss::read_scan_hits_from_capnp_file(std::string_view file_path) {
    auto in_file = detail::raii_file_for_read(file_path);

    capnp::StreamFdMessageReader message(in_file._fd);
    auto scan_reader = message.getRoot<Scan>();

    auto scan_with_hits = detail::capnp_scan_message_to_bliss_scan(scan_reader);

    return scan_with_hits;
}

void bliss::write_observation_target_hits_to_capnp_files(observation_target observation_with_hits,
                                                   std::string_view   file_path) {
    auto out_file = detail::raii_file_for_write(file_path);

    capnp::MallocMessageBuilder message;
    auto obstarget_message = message.initRoot<ObservationTarget>();

    detail::bliss_observation_target_to_capnp_observation_target_message(obstarget_message, observation_with_hits);

    capnp::writeMessageToFd(out_file._fd, message);
}

observation_target bliss::read_observation_target_hits_from_capnp_files(std::string_view file_path) {
    auto in_file = detail::raii_file_for_read(file_path);

    capnp::StreamFdMessageReader message(in_file._fd);
    auto obstarget_reader = message.getRoot<ObservationTarget>();

    auto observations = detail::capnp_observation_target_message_to_bliss_observation_target(obstarget_reader);

    return observations;
}

void bliss::write_cadence_hits_to_capnp_files(cadence cadence_with_hits, std::string_view file_path) {
    auto out_file = detail::raii_file_for_write(file_path);

    capnp::MallocMessageBuilder message;
    auto cadence_message = message.initRoot<Cadence>();

    detail::bliss_cadence_to_capnp_cadence_message(cadence_message, cadence_with_hits);

    capnp::writeMessageToFd(out_file._fd, message);
}

cadence bliss::read_cadence_hits_from_capnp_files(std::string_view file_path) {
    auto in_file = detail::raii_file_for_read(file_path);

    capnp::StreamFdMessageReader message(in_file._fd);
    auto cadence_reader = message.getRoot<Cadence>();

    auto deserialized_cadence = detail::capnp_cadence_message_to_cadence(cadence_reader);

    return deserialized_cadence;
}

#endif // BLISS_USE_CAPNP
