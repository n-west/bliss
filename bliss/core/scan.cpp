
#include "core/scan.hpp"
#include <stdexcept>

using namespace bliss;

// bliss::scan::scan() {}

bliss::scan::scan(const filterbank_data &fb_data) : filterbank_data(fb_data) {}

bliss::scan::scan(filterbank_data          fb_data,
                  bland::ndarray           dedrifted_spectrum,
                  integrated_flags         dedrifted_rfi,
                  integrate_drifts_options drift_parameters) :
        filterbank_data(fb_data),
        _dedrifted_spectrum(dedrifted_spectrum),
        _dedrifted_rfi(dedrifted_rfi),
        _drift_parameters(drift_parameters) {
    // TODO: compute/extract this somewhere more authoritative
    _integration_length = fb_data.data().size(0);
}

scan::state_tuple bliss::scan::get_state() {
    std::list<hit> hits_state = _hits.value_or(std::list<hit>{});
    noise_stats::state_tuple noise_state;
    if (_noise_stats.has_value()) {
        noise_state = _noise_stats.value().get_state();
    }
    auto state = std::make_tuple(
        filterbank_data::get_state(),
        bland::ndarray() /*doppler_spectrum*/,
        _integration_length.value_or(0),
        hits_state,
        noise_state
    );
    return state;
}

bool bliss::scan::has_doppler_spectrum() {
    return _dedrifted_spectrum.has_value();
}

bland::ndarray &bliss::scan::doppler_spectrum() {
    if (_dedrifted_spectrum.has_value()) {
        return _dedrifted_spectrum.value();
    } else {
        throw std::runtime_error("dedrifted_spectrum: have not computed dedrifted spectrum yet");
    }
}

void bliss::scan::doppler_spectrum(bland::ndarray doppler_spectrum) {
    _dedrifted_spectrum = doppler_spectrum;
}

integrated_flags &bliss::scan::doppler_flags() {
    if (_dedrifted_rfi.has_value()) {
        return _dedrifted_rfi.value();
    } else {
        throw std::runtime_error("doppler_flags (getter): have not computed dedrifted flags");
    }
}

void bliss::scan::doppler_flags(integrated_flags doppler_flags) {
    _dedrifted_rfi = doppler_flags;
}

integrate_drifts_options bliss::scan::dedoppler_options() {
    if (_drift_parameters.has_value()) {
        return _drift_parameters.value();
    } else {
        throw std::runtime_error("dedoppler_options (getter): have not computed dedrifted spectrum yet");
    }
}

void bliss::scan::dedoppler_options(integrate_drifts_options dedoppler_options) {
    _drift_parameters = dedoppler_options;
}

int64_t bliss::scan::integration_length() {
    if (_integration_length.has_value()) {
        return _integration_length.value();
    } else {
        throw std::runtime_error("_integration_length: have not computed dedrifted spectrum yet");
    }
}

void bliss::scan::integration_length(int64_t length) {
    _integration_length = length;
}

noise_stats bliss::scan::noise_estimate() {
    if (_noise_stats.has_value()) {
        return _noise_stats.value();
    } else {
        throw std::runtime_error("noise_estimate: have not computed a noise estimate yet");
    }
}

void bliss::scan::noise_estimate(noise_stats estimate) {
    _noise_stats = estimate;
}

bool bliss::scan::has_hits() {
    return _hits.has_value();
}

std::list<hit> bliss::scan::hits() {
    if (_hits.has_value()) {
        return _hits.value();
    } else {
        throw std::runtime_error("hits: have not computed hits yet");
    }
}

void bliss::scan::hits(std::list<hit> new_hits) {
    _hits = new_hits;
}
