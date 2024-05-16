#include "core/cadence.hpp"

#include <fmt/core.h>
#include <fmt/format.h>

using namespace bliss;

std::string extract_source_name_from_scans(std::vector<scan> scans) {
    std::string source_name{};
    for (auto &sc : scans) {
        if (!(sc.source_name().empty())) {
            fmt::print("INFO: Got source name {}\n", sc.source_name());
            if (source_name.empty()) {
                source_name = sc.source_name();
            } else {
                if (source_name != sc.source_name()) {
                    fmt::print("WARN: scans with different source names used to build an observation target which "
                               "expects a single source");
                    source_name += "::" + sc.source_name();
                }
            }
        }
    }
    if (source_name.empty()) {
        source_name = "unknown";
    }
    return source_name;
}

bliss::observation_target::observation_target(std::vector<scan> filterbanks) {
    for (const auto &fb : filterbanks) {
        _scans.push_back(fb);
    }
    _target_name = extract_source_name_from_scans(_scans);
}

bliss::observation_target::observation_target(std::vector<std::string> filterbank_paths, int fine_channels_per_coarse) {
    for (const auto &filterbank_path : filterbank_paths) {
        _scans.emplace_back(filterbank_path, fine_channels_per_coarse);
    }
    _target_name = extract_source_name_from_scans(_scans);
}

bool bliss::observation_target::validate_scan_consistency() {
    // Validate consistency of scans in this observation target. Each scan should:
    // * have the same number of coarse channels
    // * have the same fch1
    // * have the same foff

    double fch1;
    double foff;
    int nchans;
    if (!_scans.empty()) {
        fch1 = _scans[0].fch1();
        foff = _scans[0].foff();
        nchans = _scans[0].nchans();
    }
    for (const auto &sc : _scans) {
        // For double comparison can probably lower epsilon, but this field is
        // in MHz which gives us 1 Hz error and should be sufficient
        if ((std::abs(sc.fch1() - fch1) < 1e-6) || (std::abs(sc.foff() - foff) < 1e-6) || (sc.nchans() != nchans)) {
            fmt::print("INFO: observation_target::validate_scan_consistency: one of `fch1`, `foff`, `nchans` does not match");
            return false;
        }
    }

    return true;
}

int bliss::observation_target::get_coarse_channel_with_frequency(double frequency) {
    if (validate_scan_consistency()) {
        // Since they're consistent we can use any of them...
        return _scans[0].get_coarse_channel_with_frequency(frequency);
    } else {
        throw std::runtime_error("scans inside observation target are not consistent enough to return a channel index");
    }
}

int bliss::observation_target::get_number_coarse_channels() {
    if (validate_scan_consistency()) {
        return _scans[0].get_number_coarse_channels();
    } else {
        throw std::runtime_error("scans inside observation target are not consistent enough to return a number of channels");
    }
}

bliss::observation_target bliss::observation_target::slice_observation_channels(int start_channel, int count) {
    observation_target target_coarse_channel;
    for (auto &sc : _scans) {
        target_coarse_channel._scans.push_back(sc.slice_scan_channels(start_channel, count));
    }
    return target_coarse_channel;
}

bland::ndarray::dev bliss::observation_target::device() {
    return _device;
}

void bliss::observation_target::set_device(bland::ndarray::dev& device) {
    for (auto &target_scan : _scans) {
        target_scan.set_device(device);
    }
}

void bliss::observation_target::set_device(std::string_view dev_str) {
    bland::ndarray::dev device = dev_str;
    set_device(device);
}


bliss::cadence::cadence(std::vector<observation_target> observations) : _observations(observations) {}

bliss::cadence::cadence(std::vector<std::vector<std::string>> observations, int fine_channels_per_coarse) {
    for (const auto &target : observations) {
        _observations.emplace_back(target, fine_channels_per_coarse);
    }
}

bool bliss::cadence::validate_scan_consistency() {
    // Validate consistency of scans in this observation target. Each scan should:
    // * have the same number of coarse channels
    // * have the same fch1

    double fch1;
    double foff;
    int nchans;
    bool found_valid_scan = false;
    for (auto &obs: _observations) {
        if (!obs._scans.empty()) {
            fch1 = obs._scans[0].fch1();
            nchans = obs._scans[0].nchans();
            foff = obs._scans[0].foff();
            found_valid_scan = true;
            break;
        }
    }
    for (auto &obs: _observations) {
        for (auto & sc : obs._scans) {
            if ((std::abs(sc.fch1() - fch1) < 1e-6) || (std::abs(sc.foff() - foff) < 1e-6) || sc.nchans() != nchans) {
                fmt::print("INFO: cadence::validate_scan_consistency: one of `fch1`, `foff`, `nchans` does not match");
                return false;
            }
        }
    }
    return true;
}

int bliss::cadence::get_coarse_channel_with_frequency(double frequency) {
    if (validate_scan_consistency()) {
        // Since they're consistent we can use any of them...
        for (auto &obs: _observations) {
            if (!obs._scans.empty()) {
                return obs._scans[0].get_coarse_channel_with_frequency(frequency);
            }
        }
    }
    throw std::runtime_error("scans inside observation target are not consistent enough to return a channel index");
}

int bliss::cadence::get_number_coarse_channels() {
    if (validate_scan_consistency()) {
        // Since they're consistent we can use any of them...
        for (auto &obs: _observations) {
            if (!obs._scans.empty()) {
                return obs._scans[0].get_number_coarse_channels();
            }
        }
    }
    throw std::runtime_error("scans inside observation target are not consistent enough to return a number of channels");
}

bliss::cadence bliss::cadence::slice_cadence_channels(int start_channel, int count) {
    cadence cadence_coarse_channel;
    for (auto &obs : _observations) {
        cadence_coarse_channel._observations.push_back(obs.slice_observation_channels(start_channel, count));
    }
    return cadence_coarse_channel;
}

bland::ndarray::dev bliss::cadence::device() {
    return _device;
}

void bliss::cadence::set_device(bland::ndarray::dev& device) {
    for (auto &target : _observations) {
        target.set_device(device);
    }
}

void bliss::cadence::set_device(std::string_view dev_str) {
    bland::ndarray::dev device = dev_str;
    set_device(device);
}