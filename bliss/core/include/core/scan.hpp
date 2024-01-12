#pragma once

#include "filterbank_data.hpp"
#include "hit.hpp"

#include <bland/bland.hpp>

#include <cstdint>
#include <string>
#include <string_view>

namespace bliss {

struct integrate_drifts_options {
    // integrate_drifts_options(bool desmear, int64_t low_rate, int64_t high_rate, int64_t rate_step_size);

    // spectrum_sum_method method  = spectrum_sum_method::LINEAR_ROUND;
    bool desmear = true;
    // TODO properly optionize drift ranges
    int64_t low_rate       = -64;
    int64_t high_rate      = 64;
    int64_t rate_step_size = 1;
};

/**
 * a container to track how much rfi of each kind was involved in each drift integration
 */
struct integrated_rfi {
    bland::ndarray filter_rolloff;
    bland::ndarray low_spectral_kurtosis;
    bland::ndarray high_spectral_kurtosis;
    bland::ndarray magnitude;
    bland::ndarray sigma_clip;
    integrated_rfi(int64_t drifts, int64_t channels, bland::ndarray::dev device = default_device) :
            filter_rolloff(bland::ndarray({drifts, channels}, 0, bland::ndarray::datatype::uint8, device)),
            low_spectral_kurtosis(bland::ndarray({drifts, channels}, 0, bland::ndarray::datatype::uint8, device)),
            high_spectral_kurtosis(bland::ndarray({drifts, channels}, 0, bland::ndarray::datatype::uint8, device)),
            magnitude(bland::ndarray({drifts, channels}, 0, bland::ndarray::datatype::uint8, device)),
            sigma_clip(bland::ndarray({drifts, channels}, 0, bland::ndarray::datatype::uint8, device)) {}
};

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
    scan(filterbank_data          fb_data,
         bland::ndarray           dedrifted_spectrum,
         integrated_rfi           dedrifted_rfi,
         integrate_drifts_options drift_parameters);

    scan(const filterbank_data &fb_data);

    bland::ndarray &dedrifted_spectrum();
    integrated_rfi &dedrifted_rfi();

    integrate_drifts_options integration_options() const;
    int64_t                  integration_length() const;

    /**
     * Get the noise estimate from this scan
     */
    noise_stats noise_estimate();
    /**
     * Set a noise estimate to associate with this scan
     */
    void noise_estimate(noise_stats estimate);

    std::vector<hit> hits();
    void hits(std::vector<hit> new_hits);

  protected:
    std::optional<noise_stats> _noise_stats;

    // TODO: think through if we should wrap this up in another class that's optional rather than each one being
    // optional
    std::optional<int64_t>                  _integration_length;
    std::optional<bland::ndarray>           _dedrifted_spectrum;
    std::optional<integrated_rfi>           _dedrifted_rfi;
    std::optional<integrate_drifts_options> _drift_parameters;

    std::optional<std::vector<hit>> _hits;
};

} // namespace bliss
