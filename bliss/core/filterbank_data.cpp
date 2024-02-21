
#include "core/filterbank_data.hpp"
#include "file_types/h5_filterbank_file.hpp"

#include "fmt/format.h"
#include "fmt/ranges.h"

#include <array>

using namespace bliss;

// "The Breakthrough Listen Search for Intelligent Life: Public Data, Formats, Reduction and Archiving"
// Lebofsky, et al, 2020
// https://arxiv.org/pdf/1906.07391.pdf
// Fine channels, freq resolution (Hz), time resolution (sec), name
// clang-format off
constexpr std::array<filterbank_data::filterbank_channelization_revs, 9> known_channelizations = {{
    {1033216,      2.84, 17.98,       "HSR-Rev1A"},
    {      8, 366210.0,   0.00034953, "HTR-Rev1A"},
    {   1024,   2860.0,   1.06,       "MR-Rev1A"},

    {999424,      2.93, 17.4,        "HSR-Rev1B"},
    {     8, 366210.0,   0.00034953, "HTR-Rev1B"},
    {  1024,   2860.0,   1.02,       "MR-Rev1B"},

    {1048576,       2.79, 18.25,       "HSR-Rev2A"},
    {      8,  366210.0,   0.00034953, "HTR-Rev2A"},
    {   1024,    2860.0,   1.07,       "MR-Rev2A"}
}};
// clang-format on

std::tuple<int, filterbank_data::filterbank_channelization_revs> infer_number_coarse_channels(int number_fine_channels, double foff, double tsamp) {
    for (const auto &channelization : known_channelizations) {
        auto [fine_channels_per_coarse, freq_res, time_res, version] = channelization;

        auto num_coarse_channels = number_fine_channels / fine_channels_per_coarse;
        // Check this is an integer number of coarse channels, freq and time res are close enough
        // to expected
        if (num_coarse_channels * fine_channels_per_coarse == number_fine_channels &&
            std::abs(std::abs(foff) - freq_res) < .1 &&
            std::abs(std::abs(tsamp) - time_res) < .1) {
            return std::make_tuple(num_coarse_channels, channelization);
        }
    }
    fmt::print("WARN: filterbank_data with {} fine channels could not be matched with a known channelization scheme. "
               "Assuming 1 coarse channel\n",
               number_fine_channels);
    return {1, filterbank_data::filterbank_channelization_revs{}};
}

filterbank_data::filterbank_data(h5_filterbank_file fb_file) {
    _h5_file_handle = std::make_shared<h5_filterbank_file>(fb_file);

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

    // Find the number of coarse channels
    std::tie(_num_coarse_channels, _inferred_channelization) = infer_number_coarse_channels(_nchans, 1e6 * _foff, _tsamp);
}

filterbank_data::filterbank_data(std::string_view file_path) : filterbank_data(h5_filterbank_file(file_path)) {}

// filterbank_data::filterbank_data(bland::ndarray data, bland::ndarray mask, double foff) :
//         _data(data), _mask(mask), _foff(foff) {}

filterbank_data::filterbank_data(bland::ndarray data,
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
        // _data(data),
        // _mask(mask),
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
        _za_start(za_start) {}

filterbank_data::filterbank_data(double      fch1,
                                 double      foff,
                                 int64_t     machine_id,
                                 int64_t     nbits,
                                 int64_t     nchans,
                                 int64_t     nifs,
                                 std::string source_name,
                                 double      src_dej,
                                 double      src_raj,
                                 int64_t     telescope_id,
                                 double      tsamp,
                                 double      tstart,
                                 int64_t     data_type,
                                 double      az_start,
                                 double      za_start) :
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
    // Find the number of coarse channels
    std::tie(_num_coarse_channels, _inferred_channelization) = infer_number_coarse_channels(_nchans, 1e6 * _foff, _tsamp);
}

template <bool POPULATE_DATA_AND_MASK>
filterbank_data::state_tuple bliss::filterbank_data::get_state() {
    std::map<int, bland::ndarray> data_state;
    std::map<int, bland::ndarray> mask_state;
    if (POPULATE_DATA_AND_MASK) {
        data_state = _data;
        mask_state = _mask;
    }
    auto state = std::make_tuple(data_state,
                                 mask_state,
                                 _fch1,
                                 _foff,
                                 _machine_id,
                                 _nbits,
                                 _nchans,
                                 _nifs,
                                 _source_name,
                                 _src_dej,
                                 _src_raj,
                                 _telescope_id,
                                 _tsamp,
                                 _tstart,
                                 _data_type,
                                 _az_start,
                                 _za_start);
    return state;
}

template filterbank_data::state_tuple bliss::filterbank_data::get_state<true>();
template filterbank_data::state_tuple bliss::filterbank_data::get_state<false>();

bland::ndarray &bliss::filterbank_data::data(int coarse_channel) {
    if (coarse_channel < 0 || coarse_channel > _num_coarse_channels) {
        throw std::out_of_range("ERROR: invalid coarse channel");
    }

    if (_data.find(coarse_channel) != _data.end()) {
        return _data[coarse_channel];
    } else {
        // TODO: decide if we should evict an old coarse channel from the cache (might need
        // to stop returning references to keep that safe)

        // This is expected to be [time, feed, freq]
        auto data_count = _h5_file_handle->get_data_shape();
        data_count[2] = std::get<0>(_inferred_channelization);
        std::vector<int64_t> data_offset(3, 0);
        auto start_fine_channel = std::get<0>(_inferred_channelization) * coarse_channel;
        data_offset[2] = start_fine_channel;
        fmt::print("reading data from coarse channel {} which translates to offset {} + count {}\n", coarse_channel, data_offset, data_count);
        auto new_coarse_channel_data = _h5_file_handle->read_data(data_offset, data_count);
        _data[coarse_channel] = std::move(new_coarse_channel_data);
        return _data[coarse_channel];
    }
}

bland::ndarray &bliss::filterbank_data::mask(int coarse_channel) {
    if (_mask.find(coarse_channel) != _mask.end()) {
        return _mask[coarse_channel];
    } else {
        // TODO: check if the underlying file handle has this coarse
        // channel and read it now
        throw std::out_of_range("data does not have that coarse channel");
    }
}

int bliss::filterbank_data::get_number_coarse_channels() {
    return _num_coarse_channels;
}

double bliss::filterbank_data::fch1() {
    return _fch1;
}
void bliss::filterbank_data::fch1(double fch1) {
    _fch1 = fch1;
}

double bliss::filterbank_data::foff() {
    return _foff;
}
void bliss::filterbank_data::foff(double foff) {
    _foff = foff;
}

int64_t bliss::filterbank_data::machine_id() {
    return _machine_id;
}
void bliss::filterbank_data::machine_id(int64_t machine_id) {
    _machine_id = machine_id;
}

int64_t bliss::filterbank_data::nbits() {
    return _nbits;
}
void bliss::filterbank_data::nbits(int64_t nbits) {
    _nbits = nbits;
}

int64_t bliss::filterbank_data::nchans() {
    return _nchans;
}
void bliss::filterbank_data::nchans(int64_t nchans) {
    _nchans = nchans;
}

int64_t bliss::filterbank_data::nifs() {
    return _nifs;
}
void bliss::filterbank_data::nifs(int64_t nifs) {
    _nifs = nifs;
}

std::string bliss::filterbank_data::source_name() {
    return _source_name;
}
void bliss::filterbank_data::source_name(std::string source_name) {
    _source_name = source_name;
}

double bliss::filterbank_data::src_dej() {
    return _src_dej;
}
void bliss::filterbank_data::src_dej(double src_dej) {
    _src_dej = src_dej;
}

double bliss::filterbank_data::src_raj() {
    return _src_raj;
}
void bliss::filterbank_data::src_raj(double src_raj) {
    _src_raj = src_raj;
}

int64_t bliss::filterbank_data::telescope_id() {
    return _telescope_id;
}
void bliss::filterbank_data::telescope_id(int64_t telescope_id) {
    _telescope_id = telescope_id;
}

double bliss::filterbank_data::tsamp() {
    return _tsamp;
}
void bliss::filterbank_data::tsamp(double tsamp) {
    _tsamp = tsamp;
}

double bliss::filterbank_data::tstart() {
    return _tstart;
}
void bliss::filterbank_data::tstart(double tstart) {
    _tstart = tstart;
}

int64_t bliss::filterbank_data::data_type() {
    return _data_type;
}
void bliss::filterbank_data::data_type(int64_t data_type) {
    _data_type = data_type;
}

double bliss::filterbank_data::az_start() {
    return _az_start;
}
void bliss::filterbank_data::az_start(double az_start) {
    _az_start = az_start;
}

double bliss::filterbank_data::za_start() {
    return _za_start;
}
void bliss::filterbank_data::za_start(double za_start) {
    _za_start = za_start;
}

// void bliss::filterbank_data::mask(const bland::ndarray &mask) {
//     _mask = mask;
// }

// noise_stats bliss::filterbank_data::noise_estimates() {
//     if (_noise_stats.has_value()) {
//         return _noise_stats.value();
//     } else {
//         throw std::runtime_error("Noise stats have not been calculated yet");
//     }
// }

// void bliss::filterbank_data::noise_estimates(noise_stats stats) {
//     _noise_stats = stats;
// }
