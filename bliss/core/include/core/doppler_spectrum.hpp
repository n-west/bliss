#pragma once

#include "filterbank_data.hpp"
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

class doppler_spectrum : public filterbank_data{
  public:
    doppler_spectrum(filterbank_data          fb_data,
                     bland::ndarray           dedrifted_spectrum,
                     integrate_drifts_options drift_parameters);

    bland::ndarray &dedrifted_spectrum();

    integrate_drifts_options drift_parameters() const;

    // should we just store the options in here...
  protected:
    bland::ndarray           _dedrifted_spectrum;
    integrate_drifts_options _drift_parameters;
};

} // namespace bliss
