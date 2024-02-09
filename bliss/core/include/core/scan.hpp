#pragma once

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

    scan(filterbank_data          fb_data,
         bland::ndarray           dedrifted_spectrum,
         integrated_flags         dedrifted_rfi,
         integrate_drifts_options drift_parameters);

    using state_tuple = std::tuple<filterbank_data::state_tuple /*filterbank_data*/,
                                  bland::ndarray /*doppler_spectrum*/,
                                  int64_t /*integration_length*/,
                                  std::list<hit>,
                                  noise_stats::state_tuple>;

    state_tuple get_state();

    bool                    has_doppler_spectrum();
    bland::ndarray          &doppler_spectrum();
    void                     doppler_spectrum(bland::ndarray doppler_spectrum);
    integrated_flags        &doppler_flags();
    void                     doppler_flags(integrated_flags doppler_flags);
    integrate_drifts_options dedoppler_options();
    void                     dedoppler_options(integrate_drifts_options dedoppler_options);
    int64_t                  integration_length();
    void                     integration_length(int64_t);

    /**
     * Get the noise estimate from this scan
     */
    noise_stats noise_estimate();
    /**
     * Set a noise estimate to associate with this scan
     */
    void noise_estimate(noise_stats estimate);

    bool             has_hits();
    std::list<hit> hits();
    void             hits(std::list<hit> new_hits);

  protected:
    std::optional<noise_stats> _noise_stats;

    // TODO: think through if we should wrap this up in another class that's optional rather than each one being
    // optional
    std::optional<int64_t>                  _integration_length;
    std::optional<bland::ndarray>           _dedrifted_spectrum;
    std::optional<integrated_flags>         _dedrifted_rfi;
    std::optional<integrate_drifts_options> _drift_parameters;

    std::optional<std::list<hit>> _hits;
};

} // namespace bliss
