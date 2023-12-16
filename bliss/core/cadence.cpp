#include "core/cadence.hpp"

using namespace bliss;

bliss::observation_target::observation_target(std::vector<filterbank_data> filterbanks) : _filterbanks(filterbanks) {}

bliss::observation_target::observation_target(std::vector<std::string_view> filterbank_paths) {
    for (const auto &filterbank_path : filterbank_paths) {
        _filterbanks.emplace_back(filterbank_path);
    }
}

bliss::cadence::cadence(std::vector<observation_target> observations) : _observations(observations) {}

bliss::cadence::cadence(std::vector<std::vector<std::string_view>> observations) {
    for (const auto &target : observations) {
        _observations.emplace_back(target);
    }
}