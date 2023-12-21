#pragma once

#include "filterbank_data.hpp"

#include <string>
#include <string_view>

namespace bliss {

/**
 * Holds
 */
class observation_target {
  public:
    observation_target() = default;
    observation_target(std::vector<filterbank_data> filterbanks);
    observation_target(std::vector<std::string_view> filterbank_paths);

    // Is it useful to capture which of ABACAD this is?
    std::vector<filterbank_data> _filterbanks;
    std::string                  _target_name;
};

class cadence {
  public:
    cadence() = default;
    cadence(std::vector<observation_target> observations);
    /**
     * Read in using file paths
     */
    cadence(std::vector<std::vector<std::string_view>> observations);
    // TODO might be nice to be able to just give a list of filterbank_data, then look at that md to autosort targets

    // Is it useful to capture any data about a "primary target?"
    std::vector<observation_target> _observations;

  protected:
};
} // namespace bliss
