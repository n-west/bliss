
#if BLISS_USE_CAPNP

#include "file_types/hits_file.hpp"

#include "hit.capnp.h"

#include <errno.h>  // how to interpret errors from POSIX file
#include <fcntl.h>  // for 'open' and 'O_WRONLY'
#include <unistd.h> // for 'close'

#include <stdexcept>

#include <capnp/c++.capnp.h>
#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/exception.h>

#include <fmt/core.h>
#include <fmt/format.h>

using namespace bliss;

namespace detail {
// Fairly dumb classes to wrap up error handling to one (two) places that handle the POSIX API
// details and use RAII to make sure we don't leak resources
struct raii_file_for_write {
    int _fd;
    raii_file_for_write(std::string_view file_path) {
        _fd = open(file_path.data(), O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
        if (_fd == -1) {
            auto str_err = strerror(errno);
            auto err_msg =
                    fmt::format("write_hits_to_file: could not open file for writing (fd={}, error={})", _fd, str_err);
            throw std::runtime_error(err_msg);
        }
    }

    ~raii_file_for_write() { close(_fd); }
};
struct raii_file_for_read {
    int _fd;
    raii_file_for_read(std::string_view file_path) {
        _fd = open(file_path.data(), O_RDONLY);
        if (_fd == -1) {
            auto str_err = strerror(errno);
            auto err_msg =
                    fmt::format("write_hits_to_file: could not open file for writing (fd={}, error={})", _fd, str_err);
            throw std::runtime_error(err_msg);
        }
    }

    ~raii_file_for_read() { close(_fd); }
};

/**
 * A single source for converting our native `bliss::hit` type to a cap'n proto `signal` through a capnp builder
 */
void bliss_hit_to_capnp_signal_message(Signal::Builder &signal_builder, hit this_hit) {
    signal_builder.setFrequency(this_hit.start_freq_MHz);
    signal_builder.setIndex(this_hit.start_freq_index);
    signal_builder.setDriftSteps(
            this_hit.time_span_steps); // redundant with numTimesteps but was a rounded-up power of 2 in seticore
    signal_builder.setDriftRate(this_hit.drift_rate_Hz_per_sec);
    signal_builder.setDriftIndex(this_hit.rate_index);
    // this_hit.rate_index;
    // signal_builder.setIndex
    signal_builder.setSnr(this_hit.snr);
    // signal_builder.setCoarseChannel(); // not yet looking at multiple coarse channels
    // signal_builder.setBeam(); // not yet looking at multiple beams
    signal_builder.setNumTimesteps(this_hit.time_span_steps); // for us, redundant with driftSteps
    // the power and incoherentpower are redundant in bliss, but were not in seticore
    // the seticore power was normalized for computing an SNR, the incoherentpower was just the integrated power
    signal_builder.setPower(this_hit.power);
    signal_builder.setIncoherentPower(this_hit.power);

    // These currently aren't in the capn proto definition to be serialized
    // double  bandwidth;
    // int64_t binwidth;
    // rfi     rfi_counts;
}

hit capnp_signal_message_to_bliss_hit(const Signal::Reader &signal_reader) {
    hit this_hit;
    this_hit.start_freq_index      = signal_reader.getIndex();
    this_hit.start_freq_MHz        = signal_reader.getFrequency();
    this_hit.rate_index            = signal_reader.getDriftIndex();
    this_hit.drift_rate_Hz_per_sec = signal_reader.getDriftRate();
    this_hit.power                 = signal_reader.getPower();
    // this_hit.time_span_steps = signal_reader.getDriftSteps(); // in the cap'n proto definition, this is a
    // rounded up power-of-2 value
    this_hit.time_span_steps = signal_reader.getNumTimesteps();
    this_hit.snr             = signal_reader.getSnr();
    // The following are not (currently) in the capn proto definition
    // this_hit.bandwidth;
    // this_hit.binwidth;
    // this_hit.rfi_counts;

    return this_hit;
}
} // namespace detail

void bliss::write_hits_to_file(std::vector<hit> hits, std::string_view file_path) {

    auto out_file = detail::raii_file_for_write(file_path);

    for (size_t hit_index = 0; hit_index < hits.size(); ++hit_index) {
        auto                        this_hit = hits[hit_index];
        capnp::MallocMessageBuilder message;
        auto                        hit_builder    = message.initRoot<Hit>();
        auto                        signal_builder = hit_builder.initSignal();
        detail::bliss_hit_to_capnp_signal_message(signal_builder, this_hit);

        capnp::writeMessageToFd(out_file._fd, message);
    }
}

std::vector<hit> bliss::read_hits_from_file(std::string_view file_path) {
    auto in_file = detail::raii_file_for_read(file_path);

    std::vector<hit> hits;

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
    // fil_builder.setBeam(); // not dealing with multiple beams yet
    // fil_builder.setCoarseChannel(); // not dealing with multiple coarse channels yet
    // fil_builder.setData(); // don't need or want to duplicate large chunks of data yet
    fil_builder.setDec(scan_with_hits.src_dej());
    fil_builder.setRa(scan_with_hits.src_raj());
    fil_builder.setFoff(scan_with_hits.foff());
    // fil_builder.setNumChannels(scan_with_hits.)
    fil_builder.setSourceName(scan_with_hits.source_name());
    fil_builder.setTelescopeId(scan_with_hits.telescope_id());
    fil_builder.setTsamp(scan_with_hits.tsamp());
    fil_builder.setTstart(scan_with_hits.tstart());

    auto hits           = scan_with_hits.hits();
    auto number_hits    = hits.size();
    auto signal_builder = hit_builder.initDetections(number_hits);
    for (size_t hit_index = 0; hit_index < number_hits; ++hit_index) {
        auto this_hit    = hits[hit_index];
        auto this_signal = signal_builder[hit_index];
        detail::bliss_hit_to_capnp_signal_message(this_signal, this_hit);
    }
    auto out_file = detail::raii_file_for_write(file_path);
    capnp::writeMessageToFd(out_file._fd, message);
}

scan bliss::read_scan_hits_from_file(std::string_view file_path) {
    scan scan_with_hits;

    auto in_file = detail::raii_file_for_read(file_path);

    try {
        capnp::StreamFdMessageReader message(in_file._fd);
        auto                         hit_reader        = message.getRoot<ScanDetections>();
        auto                         deserialized_scan = hit_reader.getScan();
        scan_with_hits.fch1(deserialized_scan.getFch1());
        scan_with_hits.foff(deserialized_scan.getFoff());
        scan_with_hits.tsamp(deserialized_scan.getTsamp());
        scan_with_hits.tstart(deserialized_scan.getTstart());
        scan_with_hits.telescope_id(deserialized_scan.getTelescopeId());
        scan_with_hits.src_dej(deserialized_scan.getDec());
        scan_with_hits.src_raj(deserialized_scan.getRa());

        std::vector<hit> hits;

        auto detections  = hit_reader.getDetections();
        auto number_hits = detections.size();
        for (size_t hit_index = 0; hit_index < number_hits; ++hit_index) {
            auto this_detection = detections[hit_index];
            hit  this_hit = detail::capnp_signal_message_to_bliss_hit(this_detection);
            hits.push_back(this_hit);
        }

        scan_with_hits.hits(hits);
    } catch (kj::Exception &e) {
        // We've reached the end of the file.
        fmt::print("Error deserializing from capnp");
    }

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
    // fmt::print("Got file_paths {}\n", file_paths);
    for (auto &cadence_paths : file_paths) {
        observation_target new_target = read_observation_target_hits_from_files(cadence_paths);
        new_cadence._observations.emplace_back(new_target);
    }
    return new_cadence;
}

#endif // BLISS_USE_CAPNP
