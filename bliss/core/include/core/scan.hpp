#pragma once

#include "noise_power.hpp"
#include "filterbank_data.hpp"
#include "hit.hpp"
#include "integrate_drifts_options.hpp"
#include <bland/bland.hpp>

#include <cstdint>
#include <string>
#include <list>
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

    using state_tuple = std::tuple<filterbank_data::state_tuple /*filterbank_data*/,
                                  bland::ndarray /*doppler_spectrum*/,
                                  int64_t /*integration_length*/,
                                  std::list<hit>,
                                  noise_stats::state_tuple>;

    // state_tuple get_state();

    std::list<hit> hits();

  protected:
    std::map<int, std::shared_ptr<coarse_channel>> _coarse_channels;

};

} // namespace bliss
