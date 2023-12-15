
#include "core/doppler_spectrum.hpp"

using namespace bliss;

bliss::doppler_spectrum::doppler_spectrum(filterbank_data          fb_data,
                                          bland::ndarray           dedrifted_spectrum,
                                          integrate_drifts_options drift_parameters) :
        filterbank_data(fb_data), _dedrifted_spectrum(dedrifted_spectrum), _drift_parameters(drift_parameters) {
            // TODO: compute/extract this somewhere more authoritative
            _integration_length = fb_data.data().size(0);
        }

bland::ndarray &bliss::doppler_spectrum::dedrifted_spectrum() {
    return _dedrifted_spectrum;
}

integrate_drifts_options bliss::doppler_spectrum::integration_options() const {
    return _drift_parameters;
}

int64_t bliss::doppler_spectrum::integration_length() const {
    return _integration_length;
}
