
#include "core/coarse_channel.hpp"

#include "fmt/format.h"
#include "fmt/ranges.h"

using namespace bliss;

coarse_channel::coarse_channel(bland::ndarray data,
                                                bland::ndarray mask,
                                                double         fch1,
                                                double         foff,
                                                int64_t        machine_id,
                                                int64_t        nbits,
                                                int64_t        nchans,
                                                int64_t        nifs,
                                                std::string    source_name,
                                                double         src_dej,
                                                double         src_raj,
                                                int64_t        telescope_id,
                                                double         tsamp,
                                                double         tstart,
                                                int64_t        data_type,
                                                double         az_start,
                                                double         za_start) :
        _data(data),
        _mask(mask),
        _fch1(fch1),
        _foff(foff),
        _machine_id(machine_id),
        _nbits(nbits),
        _nchans(nchans),
        _nifs(nifs),
        _source_name(source_name),
        _src_dej(src_dej),
        _src_raj(src_raj),
        _telescope_id(telescope_id),
        _tsamp(tsamp),
        _tstart(tstart),
        _data_type(data_type),
        _az_start(az_start),
        _za_start(za_start) {
    // TODO: compute/extract this somewhere more authoritative
    _integration_length = data.size(0);
}

bland::ndarray bliss::coarse_channel::data() const {
    return _data;
}

bland::ndarray bliss::coarse_channel::mask() const {
    return _mask;
}

void bliss::coarse_channel::set_mask(bland::ndarray new_mask) {
    _mask = new_mask;
}

noise_stats bliss::coarse_channel::noise_estimate() const {
    return _noise_stats.value();
}

void bliss::coarse_channel::set_noise_estimate(noise_stats estimate) {
    _noise_stats = estimate;
}

bool bliss::coarse_channel::has_hits() {
    return _hits.has_value();
}

std::list<hit> bliss::coarse_channel::hits() const {
    return _hits.value();
}

void bliss::coarse_channel::add_hits(std::list<hit> new_hits) {
    _hits = new_hits;
}

double bliss::coarse_channel::fch1() const {
    return _fch1;
}
// void bliss::filterbank_data::fch1(double fch1) {
//     _fch1 = fch1;
// }

double bliss::coarse_channel::foff() const {
    return _foff;
}
// void bliss::filterbank_data::foff(double foff) {
//     _foff = foff;
// }

int64_t bliss::coarse_channel::machine_id() const {
    return _machine_id;
}
// void bliss::filterbank_data::machine_id(int64_t machine_id) {
//     _machine_id = machine_id;
// }

int64_t bliss::coarse_channel::nbits() const {
    return _nbits;
}
// void bliss::filterbank_data::nbits(int64_t nbits) {
//     _nbits = nbits;
// }

int64_t bliss::coarse_channel::nchans() const {
    return _nchans;
}
// void bliss::filterbank_data::nchans(int64_t nchans) {
//     _nchans = nchans;
// }

int64_t bliss::coarse_channel::nifs() const {
    return _nifs;
}
// void bliss::filterbank_data::nifs(int64_t nifs) {
//     _nifs = nifs;
// }

std::string bliss::coarse_channel::source_name() const {
    return _source_name;
}
// void bliss::filterbank_data::source_name(std::string source_name) {
//     _source_name = source_name;
// }

double bliss::coarse_channel::src_dej() const {
    return _src_dej;
}
// void bliss::filterbank_data::src_dej(double src_dej) {
//     _src_dej = src_dej;
// }

double bliss::coarse_channel::src_raj() const {
    return _src_raj;
}
// void bliss::filterbank_data::src_raj(double src_raj) {
//     _src_raj = src_raj;
// }

int64_t bliss::coarse_channel::telescope_id() const {
    return _telescope_id;
}
// void bliss::filterbank_data::telescope_id(int64_t telescope_id) {
//     _telescope_id = telescope_id;
// }

double bliss::coarse_channel::tsamp() const {
    return _tsamp;
}
// void bliss::filterbank_data::tsamp(double tsamp) {
//     _tsamp = tsamp;
// }

double bliss::coarse_channel::tstart() const {
    return _tstart;
}
// void bliss::filterbank_data::tstart(double tstart) {
//     _tstart = tstart;
// }

int64_t bliss::coarse_channel::data_type() const {
    return _data_type;
}
// void bliss::filterbank_data::data_type(int64_t data_type) {
//     _data_type = data_type;
// }

double bliss::coarse_channel::az_start() const {
    return _az_start;
}
// void bliss::filterbank_data::az_start(double az_start) {
//     _az_start = az_start;
// }

double bliss::coarse_channel::za_start() const {
    return _za_start;
}
// void bliss::filterbank_data::za_start(double za_start) {
//     _za_start = za_start;
// }

bool bliss::coarse_channel::has_doppler_spectrum() {
    return _dedrifted_spectrum.has_value();
}

bland::ndarray &bliss::coarse_channel::doppler_spectrum() {
    if (_dedrifted_spectrum.has_value()) {
        return _dedrifted_spectrum.value();
    } else {
        throw std::runtime_error("dedrifted_spectrum: have not computed dedrifted spectrum yet");
    }
}

void bliss::coarse_channel::doppler_spectrum(bland::ndarray doppler_spectrum) {
    _dedrifted_spectrum = doppler_spectrum;
}

integrated_flags &bliss::coarse_channel::doppler_flags() {
    if (_dedrifted_rfi.has_value()) {
        return _dedrifted_rfi.value();
    } else {
        throw std::runtime_error("doppler_flags (getter): have not computed dedrifted flags");
    }
}

void bliss::coarse_channel::doppler_flags(integrated_flags doppler_flags) {
    _dedrifted_rfi = doppler_flags;
}

integrate_drifts_options bliss::coarse_channel::dedoppler_options() {
    if (_drift_parameters.has_value()) {
        return _drift_parameters.value();
    } else {
        throw std::runtime_error("dedoppler_options (getter): have not computed dedrifted spectrum yet");
    }
}

void bliss::coarse_channel::dedoppler_options(integrate_drifts_options dedoppler_options) {
    _drift_parameters = dedoppler_options;
}

int64_t bliss::coarse_channel::integration_length() {
    if (_integration_length.has_value()) {
        return _integration_length.value();
    } else {
        throw std::runtime_error("_integration_length: have not computed dedrifted spectrum yet");
    }
}

void bliss::coarse_channel::integration_length(int64_t length) {
    _integration_length = length;
}
