
#include "core/doppler_spectrum.hpp"

using namespace bliss;

// bliss::integrate_drifts_options::integrate_drifts_options(bool    desmear,
//                                                           int64_t low_rate,
//                                                           int64_t high_rate,
//                                                           int64_t rate_step_size) :
//         desmear(desmear), low_rate(low_rate), high_rate(high_rate), rate_step_size(rate_step_size) {}

bliss::doppler_spectrum::doppler_spectrum(filterbank_data          fb_data,
                                          bland::ndarray           dedrifted_spectrum,
                                          integrate_drifts_options drift_parameters) :
         filterbank_data(fb_data), _dedrifted_spectrum(dedrifted_spectrum), _drift_parameters(drift_parameters) {}

bland::ndarray &bliss::doppler_spectrum::dedrifted_spectrum() {
    return _dedrifted_spectrum;
}

integrate_drifts_options bliss::doppler_spectrum::drift_parameters() const {
    return _drift_parameters;
}