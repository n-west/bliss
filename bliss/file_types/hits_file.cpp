
#if BLISS_USE_CAPNP

#include "file_types/hits_file.hpp"
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
void bliss::write_hits_to_file(Container<hit> hits, std::string_view file_path) {

    auto out_file = detail::raii_file_for_write(file_path);

    for (auto &this_hit : hits) {
        capnp::MallocMessageBuilder message;
        auto                        hit_builder    = message.initRoot<Hit>();
        auto                        signal_builder = hit_builder.initSignal();
        detail::bliss_hit_to_capnp_signal_message(signal_builder, this_hit);

        capnp::writeMessageToFd(out_file._fd, message);
    }
}
template void bliss::write_hits_to_file<std::vector>(std::vector<hit> hits, std::string_view file_path);
template void bliss::write_hits_to_file<std::list>(std::list<hit> hits, std::string_view file_path);


std::list<hit> bliss::read_hits_from_file(std::string_view file_path) {
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

void bliss::write_scan_hits_to_file(scan scan_with_hits, std::string_view file_path) {
    capnp::MallocMessageBuilder message;
    auto                        hit_builder = message.initRoot<ScanDetections>();
    auto                        fil_builder = hit_builder.initScan();
    fil_builder.setFch1(scan_with_hits.fch1());
    // fil_builder.setBeam(); // not dealing with multiple beams (yet)
    // fil_builder.setCoarseChannel(); // not dealing with multiple coarse channels (yet)
    // fil_builder.setData(); // don't need or want to duplicate large chunks of data (yet)
    fil_builder.setDec(scan_with_hits.src_dej());
    fil_builder.setRa(scan_with_hits.src_raj());
    fil_builder.setFoff(scan_with_hits.foff());
    // fil_builder.setNumChannels(scan_with_hits.)
    fil_builder.setSourceName(scan_with_hits.source_name());
    fil_builder.setTelescopeId(scan_with_hits.telescope_id());
    fil_builder.setTsamp(scan_with_hits.tsamp());
    fil_builder.setTstart(scan_with_hits.tstart());
    fil_builder.setNumTimesteps(scan_with_hits.slow_time_bins());
    fil_builder.setNumChannels(scan_with_hits.nchans());

    auto hits           = scan_with_hits.hits();
    auto number_hits    = hits.size();
    auto signal_builder = hit_builder.initDetections(number_hits);
    size_t hit_index = 0;
    for (const auto &hit : hits) {
        // auto this_hit    = hits[hit_index];
        auto this_signal = signal_builder[hit_index];
        detail::bliss_hit_to_capnp_signal_message(this_signal, hit);
        ++hit_index;
    }
    fmt::print("Have {} hits (expected {})\n", hit_index, number_hits);
    auto out_file = detail::raii_file_for_write(file_path);
    capnp::writeMessageToFd(out_file._fd, message);
}

scan bliss::read_scan_hits_from_file(std::string_view file_path) {
    scan scan_with_hits;

    auto in_file = detail::raii_file_for_read(file_path);

    try {
        capnp::StreamFdMessageReader message(in_file._fd);

        auto hit_reader        = message.getRoot<ScanDetections>();
        auto deserialized_scan = hit_reader.getScan();
        fmt::print("Setting fields of scan\n");
        auto fch1 = deserialized_scan.getFch1();
        scan_with_hits.set_fch1(fch1);
        fmt::print("made it here\n");
        scan_with_hits.set_foff(deserialized_scan.getFoff());
        scan_with_hits.set_tsamp(deserialized_scan.getTsamp());
        scan_with_hits.set_tstart(deserialized_scan.getTstart());
        scan_with_hits.set_telescope_id(deserialized_scan.getTelescopeId());
        scan_with_hits.set_src_dej(deserialized_scan.getDec());
        scan_with_hits.set_src_raj(deserialized_scan.getRa());
        scan_with_hits.set_nchans(deserialized_scan.getNumChannels());
        fmt::print("Done setting fields\n");

        std::list<hit> hits;

        auto detections  = hit_reader.getDetections();
        auto number_hits = detections.size();
        fmt::print("number hits: {}\n", number_hits);
        for (size_t hit_index = 0; hit_index < number_hits; ++hit_index) {
            fmt::print("pushing back hit index {}/{}\n", hit_index, number_hits);

            auto this_detection = detections[hit_index];
            hit  this_hit = detail::capnp_signal_message_to_bliss_hit(this_detection);
            hits.push_back(this_hit);
        }

        // TODO: fix this
        // scan_with_hits.hits(hits);
    } catch (kj::Exception &e) {
        fmt::print("{}\n", e.getDescription().cStr());
        // We've reached the end of the file.
    } catch (std::exception e) {
        fmt::print("{}\n", e.what());
    }

    fmt::print("returning scan\n");

    return scan_with_hits;
}

void bliss::write_observation_target_hits_to_files(observation_target observation_with_hits,
                                                   std::string_view   base_filename) {
    for (size_t scan_index = 0; scan_index < observation_with_hits._scans.size(); ++scan_index) {
        // TODO: is there a way to verify if the given file_path has a proper format slot availability?
        auto path_for_this_scan = fmt::format("{}_{}.cp", base_filename, scan_index);
        write_scan_hits_to_file(observation_with_hits._scans[scan_index], path_for_this_scan);
    }
}

observation_target bliss::read_observation_target_hits_from_files(std::vector<std::string_view> file_paths) {
    observation_target observations;
    for (auto &file_path : file_paths) {
        auto new_scan = read_scan_hits_from_file(file_path);
        observations._scans.emplace_back(new_scan);
    }

    return observations;
}

void bliss::write_cadence_hits_to_files(cadence cadence_with_hits, std::string_view base_filename) {
    for (size_t target_index = 0; target_index < cadence_with_hits._observations.size(); ++target_index) {
        auto target                = cadence_with_hits._observations[target_index];
        auto formatted_target_base = fmt::format("{}_obs{}-{}", base_filename, target_index, target._target_name);
        write_observation_target_hits_to_files(target, formatted_target_base);
    }
}

cadence bliss::read_cadence_hits_from_files(std::vector<std::vector<std::string_view>> file_paths) {
    cadence new_cadence;
    for (auto &cadence_paths : file_paths) {
        observation_target new_target = read_observation_target_hits_from_files(cadence_paths);
        new_cadence._observations.emplace_back(new_target);
    }
    return new_cadence;
}

#endif // BLISS_USE_CAPNP
