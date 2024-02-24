#pragma once

#include "filterbank_data.hpp"
#include "hit.hpp"
#include "integrate_drifts_options.hpp"
#include "noise_power.hpp"
#include <bland/bland.hpp>

#include <cstdint>
#include <list>
#include <string>
#include <string_view>

namespace bliss {

/**
 * scan holds the original filterbank_data and directly derived products that are only useful when attached to the
 * underlying filterbank
 *
 * These derived products are:
 * * dedrifted_spectrum
 * * dedrifted_flags
 */
class scan : public filterbank_data {
  public:
    scan() = default;

    scan(const filterbank_data &fb_data);

    // scan(const scan &scan_data);

    // using state_tuple = std::tuple<filterbank_data::state_tuple /*filterbank_data*/,
    //                                bland::ndarray /*doppler_spectrum*/,
    //                                int64_t /*integration_length*/,
    //                                std::list<hit>,
    //                                noise_stats::state_tuple>;

    // state_tuple get_state();

    /**
     * gather hits in all coarse channels of this scan and return as a single list
     */
    std::list<hit> hits();

    /**
     * create a new scan consisting of the selected coarse channel
     */
    scan extract_coarse_channels(int start_channel = 0, int count = 1);

  protected:
    // std::map<int, std::shared_ptr<coarse_channel>> _coarse_channels;

};

} // namespace bliss
