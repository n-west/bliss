#pragma once

#include "scan.hpp"

#include <string>
#include <string_view>

namespace bliss {

/**
 * An observation target is a physical object / location in the sky that is observed. It may have multiple
 * scans (different observations). These are held as `scan` objects which hold the underlying `filterbank_data`
 * as well as optionally derived data products that are directly tied to filterbank_data.
 */
struct observation_target {
  public:
    observation_target() = default;
    observation_target(std::vector<filterbank_data> filterbanks);
    observation_target(std::vector<scan> filterbanks);
    observation_target(std::vector<std::string_view> filterbank_paths);

    // Is it useful to capture which of ABACAD this is?
    std::vector<scan> _scans;
    std::string       _target_name;
};

/**
 * A cadence collects multiple observation targets from a single observing run. A common procedure for single-dish
 * telescopes is to run an ABACAD cadence where A is the primary target of interest (some exoplanet or star) and B, C, D
 * are different objects or empty sky for the purpose of finding signals which only come from the A observation.
 */
struct cadence {
  public:
    cadence() = default;

    /**
     * Build a cadence from observation_targets
    */
    cadence(std::vector<observation_target> observations);

    /**
     * Build a cadence by reading file paths to scans
     */
    cadence(std::vector<std::vector<std::string_view>> observations);
    // TODO might be nice to be able to just give a list of filterbank_data, then look at that metadata to autosort
    // targets

    // Is it useful to capture any data about a "primary target?"
    std::vector<observation_target> _observations;

  protected:
};
} // namespace bliss
