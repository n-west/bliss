#include "core/cadence.hpp"

using namespace bliss;

bliss::observation_target::observation_target(std::vector<filterbank_data> filterbanks) {
    for (const auto &fb : filterbanks) {
        _scans.push_back(fb);
    }
}

bliss::observation_target::observation_target(std::vector<scan> scans) : _scans(scans) {}

bliss::observation_target::observation_target(std::vector<std::string_view> filterbank_paths) {
    for (const auto &filterbank_path : filterbank_paths) {
        _scans.emplace_back(filterbank_path);
    }
}

bliss::cadence::cadence(std::vector<observation_target> observations) : _observations(observations) {}

bliss::cadence::cadence(std::vector<std::vector<std::string_view>> observations) {
    for (const auto &target : observations) {
        _observations.emplace_back(target);
    }
}