
#include "core/filterbank_data.hpp"
#include "file_types/h5_filterbank_file.hpp"

#include "fmt/format.h"
#include <bland/bland.hpp>

using namespace bliss;

filterbank_data::filterbank_data(h5_filterbank_file fb_file) :
        _data(fb_file.read_data()), _mask(fb_file.read_mask()) {

    // double      fch1;
    _fch1 = fb_file.read_data_attr<double>("fch1");
    // double      foff;
    _foff = fb_file.read_data_attr<double>("foff");
    // int64_t     machine_id;
    _machine_id = fb_file.read_data_attr<int64_t>("machine_id");
    // int64_t     nbits;
    _nbits = fb_file.read_data_attr<int64_t>("nbits");
    // int64_t     nchans;
    _nchans = fb_file.read_data_attr<int64_t>("nchans");
    // int64_t     nifs;
    _nifs = fb_file.read_data_attr<int64_t>("nifs");
    // std::string source_name;
    _source_name = fb_file.read_data_attr<std::string>("source_name");
    // double      src_dej;
    _src_dej = fb_file.read_data_attr<double>("src_dej");
    // double      src_raj;
    _src_raj = fb_file.read_data_attr<double>("src_raj");
    // int64_t     telescope_id;
    _telescope_id = fb_file.read_data_attr<int64_t>("telescope_id");
    // double      tstamp;
    _tsamp = fb_file.read_data_attr<double>("tsamp");
    // double      tstart;
    _tstart = fb_file.read_data_attr<double>("tstart");

    // int64_t data_type;
    _data_type = fb_file.read_data_attr<int64_t>("data_type");
    // double  az_start;
    _az_start = fb_file.read_data_attr<double>("az_start");
    // double  za_start;
    _za_start = fb_file.read_data_attr<double>("za_start");
}

filterbank_data::filterbank_data(std::string_view file_path) : filterbank_data(h5_filterbank_file(file_path)) {}

filterbank_data::filterbank_data(bland::ndarray data, bland::ndarray mask, double foff) : _data(data), _mask(mask), _foff(foff) {}

bland::ndarray& bliss::filterbank_data::data() {
    return _data;
}

bland::ndarray& bliss::filterbank_data::mask() {
    return _mask;
}

double bliss::filterbank_data::fch1() const {
    return _fch1;
}
double bliss::filterbank_data::foff() const {
    return _foff;
}
int64_t bliss::filterbank_data::machine_id() const {
    return _machine_id;
}
int64_t bliss::filterbank_data::nbits() const {
    return _nbits;
}
int64_t bliss::filterbank_data::nchans() const {
    return _nchans;
}
int64_t bliss::filterbank_data::nifs() const {
    return _nifs;
}
std::string bliss::filterbank_data::source_name() const {
    return _source_name;
}
double bliss::filterbank_data::src_dej() const {
    return _src_dej;
}
double bliss::filterbank_data::src_raj() const {
    return _src_raj;
}
int64_t bliss::filterbank_data::telescope_id() const {
    return _telescope_id;
}
double bliss::filterbank_data::tsamp() const {
    return _tsamp;
}
double bliss::filterbank_data::tstart() const {
    return _tstart;
}

int64_t bliss::filterbank_data::data_type() const {
    return _data_type;
}
double bliss::filterbank_data::az_start() const {
    return _az_start;
}
double bliss::filterbank_data::za_start() const {
    return _za_start;
}
