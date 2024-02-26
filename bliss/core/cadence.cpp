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

bliss::observation_target::observation_target(std::vector<std::string_view> filterbank_paths) {
    for (const auto &filterbank_path : filterbank_paths) {
        _scans.emplace_back(filterbank_path);
    }
    _target_name = extract_source_name_from_scans(_scans);
}

bliss::observation_target bliss::observation_target::slice_observation_channels(int start_channel, int count) {
    observation_target target_coarse_channel;
    for (auto &sc : _scans) {
        target_coarse_channel._scans.push_back(sc.slice_observation_channels(start_channel, count));
    }
    return target_coarse_channel;
}

bliss::cadence::cadence(std::vector<observation_target> observations) : _observations(observations) {}

bliss::cadence::cadence(std::vector<std::vector<std::string_view>> observations) {
    for (const auto &target : observations) {
        _observations.emplace_back(target);
    }
}

bliss::cadence bliss::cadence::slice_cadence_channels(int start_channel, int count) {
    cadence cadence_coarse_channel;
    for (auto &obs : _observations) {
        cadence_coarse_channel._observations.push_back(obs.slice_observation_channels(start_channel, count));
    }
    return cadence_coarse_channel;
}
