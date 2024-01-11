
#include "core/scan.hpp"
#include <stdexcept>

using namespace bliss;

bliss::scan::scan(filterbank_data          fb_data,
                                          bland::ndarray           dedrifted_spectrum,
                                          integrated_rfi           dedrifted_rfi,
                                          integrate_drifts_options drift_parameters) :
        filterbank_data(fb_data),
        _dedrifted_spectrum(dedrifted_spectrum),
        _dedrifted_rfi(dedrifted_rfi),
        _drift_parameters(drift_parameters) {
    // TODO: compute/extract this somewhere more authoritative
    _integration_length = fb_data.data().size(0);
}

bliss::scan::scan(const filterbank_data &fb_data) : filterbank_data(fb_data) {
}

bland::ndarray &bliss::scan::dedrifted_spectrum() {
    if (_dedrifted_spectrum.has_value()) {
        return _dedrifted_spectrum.value();
    } else {
        throw std::runtime_error("dedrifted_spectrum: have not computed dedrifted spectrum yet");
    }
}

integrated_rfi &bliss::scan::dedrifted_rfi() {
    if (_dedrifted_rfi.has_value()) {
        return _dedrifted_rfi.value();
    } else {
        throw std::runtime_error("_dedrifted_rfi: have not computed dedrifted spectrum yet");
    }
}

integrate_drifts_options bliss::scan::integration_options() const {
    if (_drift_parameters.has_value()) {
        return _drift_parameters.value();
    } else {
        throw std::runtime_error("_drift_parameters: have not computed dedrifted spectrum yet");
    }
}

int64_t bliss::scan::integration_length() const {
    if (_integration_length.has_value()) {
        return _integration_length.value();
    } else {
        throw std::runtime_error("_integration_length: have not computed dedrifted spectrum yet");
    }
}
