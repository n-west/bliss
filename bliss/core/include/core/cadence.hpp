#pragma once

#include "scan.hpp"

#include <string>
#include <string_view>

namespace bliss {

/**
 * An observation target is a physical object / location in the sky that is observed. It may have multiple
 * scans (different observations). These are held as `scan` objects which hold the underlying `scan`
 * as well as optionally derived data products that are directly tied to scan.
 */
struct observation_target {
  public:
    observation_target() = default;
    observation_target(std::vector<scan> filterbanks);

    /**
     * Create an observation target from a list of scan filepaths and a number of fine channels per
     * coarse channel to assume each scan has.
    */
    observation_target(std::vector<std::string> filterbank_paths, int fine_channels_per_coarse=0);

    /**
     * Validate consistency of scans in this observation target. Each scan should:
     * * have the same number of coarse channels
     * * have the same fch1
     * * have the same foff
    */
    bool validate_scan_consistency();

    /**
     * return the coarse channel index that the given frequency is in
     * 
     * useful for reinvestigating hits by looking up frequency
     * 
     * In the future this may change name or return the actual coarse channel
    */
    int get_coarse_channel_with_frequency(double frequency);

    /**
     * get the number of coarse channels in underlying filterbanks (per scan)
     *
     * This value is derived from known channelizations of BL backends and
     * the actual number of fine channels in this filterbank
     */
    int get_number_coarse_channels();

    /**
     * create a new observation_target consisting of a slice of coarse channels
    */
    observation_target slice_observation_channels(int start_channel=0, int count=1);

    bland::ndarray::dev device();
    void set_device(bland::ndarray::dev &device);
    void set_device(std::string_view device_str);

    // Is it useful to capture which of ABACAD this is?
    std::vector<scan> _scans;
    std::string       _target_name;

  protected:
    bland::ndarray::dev _device = default_device;
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
     * Build a cadence by reading file paths to scans assuming the given number of fine channels per coarse
     * in each scan
    */
    cadence(std::vector<std::vector<std::string>> observations, int fine_channels_per_coarse=0);
    // TODO might be nice to be able to just give a list of scan, then look at that metadata to autosort
    // targets

    // Is it useful to capture any data about a "primary target?"
    std::vector<observation_target> _observations;

    /**
     * Validate consistency of scans in this cadence. Each scan should:
     * * have the same number of coarse channels
     * * have the same fch1
     * * have same foff
    */
    bool validate_scan_consistency();

    /**
     * return the coarse channel index that the given frequency is in
     *
     * useful for reinvestigating hits by looking up frequency
     *
     * In the future this may change name or return the actual coarse channel
    */
    int get_coarse_channel_with_frequency(double frequency);

    /**
     * get the number of coarse channels in underlying filterbanks (per scan)
     *
     * This value is derived from known channelizations of BL backends and
     * the actual number of fine channels in this filterbank
     */
    int get_number_coarse_channels();

    /**
     * create a new cadence consisting of a slice of coarse channels
    */
    cadence slice_cadence_channels(int start_channel=0, int count=1);

    bland::ndarray::dev device();
    void set_device(bland::ndarray::dev &device);
    void set_device(std::string_view device_str);

  protected:
    bland::ndarray::dev _device = default_device;
};
} // namespace bliss
