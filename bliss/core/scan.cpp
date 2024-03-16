
#include "core/scan.hpp"
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
constexpr std::array<scan::filterbank_channelization_revs, 9> known_channelizations = {{
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

std::tuple<int, scan::filterbank_channelization_revs>
infer_number_coarse_channels(int number_fine_channels, double foff, double tsamp) {
    for (const auto &channelization : known_channelizations) {
        auto [fine_channels_per_coarse, freq_res, time_res, version] = channelization;

        auto num_coarse_channels = number_fine_channels / fine_channels_per_coarse;
        // Check this is an integer number of coarse channels, freq and time res are close enough
        // to expected
        if (num_coarse_channels * fine_channels_per_coarse == number_fine_channels &&
            std::abs(std::abs(foff) - freq_res) < .1 && std::abs(std::abs(tsamp) - time_res) < .1) {
            return std::make_tuple(num_coarse_channels, channelization);
        }
    }
    fmt::print("WARN: scan with {} fine channels could not be matched with a known channelization scheme. "
               "Assuming 1 coarse channel\n",
               number_fine_channels);
    return {1, scan::filterbank_channelization_revs{}};
}

scan::scan(h5_filterbank_file fb_file) {
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
    std::tie(_num_coarse_channels, _inferred_channelization) =
            infer_number_coarse_channels(_nchans, 1e6 * _foff, _tsamp);
}

scan::scan(std::string_view file_path) : scan(h5_filterbank_file(file_path)) {}

scan::scan(bland::ndarray data,
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

scan::scan(double      fch1,
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
    std::tie(_num_coarse_channels, _inferred_channelization) =
            infer_number_coarse_channels(_nchans, 1e6 * _foff, _tsamp);

    auto shape      = _h5_file_handle->get_data_shape(); // should we just hold on to the shape?
    _slow_time_bins = shape[0];
    // This assumes there is no overlap which needs to be confirmed with data paper
    _tduration_secs = _slow_time_bins * _tsamp;
}

template <bool POPULATE_DATA_AND_MASK>
scan::state_tuple bliss::scan::get_state() {
    std::map<int, bland::ndarray> data_state;
    std::map<int, bland::ndarray> mask_state;
    if (POPULATE_DATA_AND_MASK) {
        // data_state = _data;
        // mask_state = _mask;
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

template scan::state_tuple bliss::scan::get_state<true>();
template scan::state_tuple bliss::scan::get_state<false>();

std::shared_ptr<coarse_channel> bliss::scan::get_coarse_channel(int coarse_channel_index) {
    if (coarse_channel_index < 0 || coarse_channel_index > _num_coarse_channels) {
        throw std::out_of_range("ERROR: invalid coarse channel");
    }

    auto global_offset_in_file = coarse_channel_index + _coarse_channel_offset;
    if (_coarse_channels.find(global_offset_in_file) == _coarse_channels.end()) {
        // TODO: decide if we should evict an old coarse channel from the cache (might need
        // to stop returning references to keep that safe)

        // This is expected to be [time, feed, freq]
        auto data_count = _h5_file_handle->get_data_shape();
        std::vector<int64_t> data_offset(3, 0);

        data_count[2] = std::get<0>(_inferred_channelization);
        auto global_start_fine_channel = std::get<0>(_inferred_channelization) * global_offset_in_file;
        data_offset[2] = global_start_fine_channel;

        fmt::print("DEBUG: reading data from coarse channel {} which translates to offset {} + count {}\n",
                   global_offset_in_file,
                   data_offset,
                   data_count);
        auto data_reader = [h5_file_handle=this->_h5_file_handle, data_offset, data_count](){
            return h5_file_handle->read_data(data_offset, data_count);
        };
        auto mask_reader = [h5_file_handle=this->_h5_file_handle, data_offset, data_count](){
            return h5_file_handle->read_mask(data_offset, data_count);
        };
        
        // auto new_coarse_channel_data = _h5_file_handle->read_data(data_offset, data_count);
        // auto new_coarse_channel_mask = _h5_file_handle->read_mask(data_offset, data_count);
        auto relative_start_fine_channel = std::get<0>(_inferred_channelization) * coarse_channel_index;
        auto coarse_fch1             = _fch1 + _foff * relative_start_fine_channel;

        auto new_coarse = std::make_shared<coarse_channel>(data_reader,
                                                           mask_reader,
                                                           coarse_fch1,
                                                           _foff,
                                                           _machine_id,
                                                           _nbits,
                                                           std::get<0>(_inferred_channelization),
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
        new_coarse->set_device(_device);
        _coarse_channels.insert({global_offset_in_file, new_coarse});
    }
    auto cc = _coarse_channels.at(global_offset_in_file);
    cc->set_device(_device);
    return cc;
}

bliss::scan::filterbank_channelization_revs bliss::scan::get_channelization() {
    return _inferred_channelization;
}

int bliss::scan::get_coarse_channel_with_frequency(double frequency) {
    auto band_fraction = ((frequency - _fch1) / _foff / static_cast<double>(_nchans));
    // TODO: if band_fraction is < 0 or > 1 then it's not in this filterbank. Throw an error
    auto fractional_channel = band_fraction * _num_coarse_channels;
    return std::floor(fractional_channel);
}

int bliss::scan::get_number_coarse_channels() {
    return _num_coarse_channels;
}

std::list<hit> bliss::scan::hits() {
    std::list<hit> all_hits;
    int            number_coarse_channels = get_number_coarse_channels();
    for (int cc_index = 0; cc_index < number_coarse_channels; ++cc_index) {
        auto cc = get_coarse_channel(cc_index);
        try {
            auto this_channel_hits = cc->hits();
            all_hits.insert(all_hits.end(), this_channel_hits.cbegin(), this_channel_hits.cend());
        } catch (const std::bad_optional_access &e) {
            fmt::print("WARN: no hits available from coarse channel {}\n", cc_index);
            // TODO: catch specific exception we know might be thrown
        }
    }
    return all_hits;
}


bland::ndarray::dev bliss::scan::device() {
    return _device;
}

void bliss::scan::set_device(bland::ndarray::dev &device) {
    _device = device;
    for (auto &[channel_index, cc] : _coarse_channels) {
        cc->set_device(device);
    }
}

void bliss::scan::set_device(std::string_view dev_str) {
    bland::ndarray::dev device = dev_str;
    set_device(device);
}

bliss::scan bliss::scan::slice_scan_channels(int start_channel, int count) {
    auto sliced_scan = *this;

    // what are implications of negative numbers here?
    sliced_scan._coarse_channel_offset += start_channel;
    sliced_scan._num_coarse_channels = count;

    int fine_channels_per_coarse = std::get<0>(_inferred_channelization);

    sliced_scan._fch1 = _fch1 + _foff * fine_channels_per_coarse * start_channel;
    sliced_scan._nchans = count * fine_channels_per_coarse;

    return sliced_scan;
}

double bliss::scan::fch1() const {
    return _fch1;
}
void bliss::scan::set_fch1(double fch1) {
    _fch1 = fch1;
}

double bliss::scan::foff() const {
    return _foff;
}
void bliss::scan::set_foff(double foff) {
    _foff = foff;
}

int64_t bliss::scan::machine_id() const {
    return _machine_id;
}
void bliss::scan::set_machine_id(int64_t machine_id) {
    _machine_id = machine_id;
}

int64_t bliss::scan::nbits() const {
    return _nbits;
}
void bliss::scan::set_nbits(int64_t nbits) {
    _nbits = nbits;
}

int64_t bliss::scan::nchans() const {
    return _nchans;
}
void bliss::scan::set_nchans(int64_t nchans) {
    _nchans = nchans;
}

int64_t bliss::scan::nifs() const {
    return _nifs;
}
void bliss::scan::set_nifs(int64_t nifs) {
    _nifs = nifs;
}

std::string bliss::scan::source_name() const {
    return _source_name;
}
void bliss::scan::set_source_name(std::string source_name) {
    _source_name = source_name;
}

double bliss::scan::src_dej() const {
    return _src_dej;
}
void bliss::scan::set_src_dej(double src_dej) {
    _src_dej = src_dej;
}

double bliss::scan::src_raj() const {
    return _src_raj;
}
void bliss::scan::set_src_raj(double src_raj) {
    _src_raj = src_raj;
}

int64_t bliss::scan::telescope_id() const {
    return _telescope_id;
}
void bliss::scan::set_telescope_id(int64_t telescope_id) {
    _telescope_id = telescope_id;
}

double bliss::scan::tsamp() const {
    return _tsamp;
}
void bliss::scan::set_tsamp(double tsamp) {
    _tsamp = tsamp;
}

double bliss::scan::tstart() const {
    return _tstart;
}
void bliss::scan::set_tstart(double tstart) {
    _tstart = tstart;
}

int64_t bliss::scan::data_type() const {
    return _data_type;
}
void bliss::scan::set_data_type(int64_t data_type) {
    _data_type = data_type;
}

double bliss::scan::az_start() const {
    return _az_start;
}
void bliss::scan::set_az_start(double az_start) {
    _az_start = az_start;
}

double bliss::scan::za_start() const {
    return _za_start;
}
void bliss::scan::set_za_start(double za_start) {
    _za_start = za_start;
}

int64_t bliss::scan::slow_time_bins() const {
    return _slow_time_bins;
}

double bliss::scan::tduration_secs() const {
    return _tduration_secs;
}
