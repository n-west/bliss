#pragma once

#include "coarse_channel.hpp"

#include <bland/bland.hpp>

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>

namespace bliss {

struct h5_filterbank_file;

class filterbank_data {
  public:
    filterbank_data() = default;
    filterbank_data(h5_filterbank_file fb_file);
    // filterbank_data(bland::ndarray data, bland::ndarray mask, double foff = 1);
    filterbank_data(bland::ndarray data,
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
                    double         za_start);
    filterbank_data(double      fch1,
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
                    double      za_start);
    filterbank_data(std::string_view file_path);

    using state_tuple = std::tuple<std::map<int, bland::ndarray>,
                                   std::map<int, bland::ndarray>,
                                   double,
                                   double,
                                   int64_t,
                                   int64_t,
                                   int64_t,
                                   int64_t,
                                   std::string,
                                   double,
                                   double,
                                   int64_t,
                                   double,
                                   double,
                                   int64_t,
                                   double,
                                   double>;

    template <bool POPULATE_DATA_AND_MASK = false>
    state_tuple get_state();

    std::shared_ptr<coarse_channel> get_coarse_channel(int coarse_channel_index=0);

    // bland::ndarray data(int coarse_channel_index=0);
    // bland::ndarray mask(int coarse_channel_index=0);

    /**
     * get the number of coarse channels in this filterbank
     * 
     * This value is derived from known channelizations of BL backends and
     * the actual number of fine channels in this filterbank
    */
    int get_number_coarse_channels();
    using filterbank_channelization_revs = std::tuple<int, double, double, const char*>;

    // Setters and getters for values read from disk
    double      fch1() const;
    void        set_fch1(double);
    double      foff() const;
    void        set_foff(double);
    int64_t     machine_id() const;
    void        set_machine_id(int64_t);
    int64_t     nbits() const;
    void        set_nbits(int64_t);
    int64_t     nchans() const;
    void        set_nchans(int64_t);
    int64_t     nifs() const;
    void        set_nifs(int64_t);
    std::string source_name() const;
    void        set_source_name(std::string);
    double      src_dej() const;
    void        set_src_dej(double);
    double      src_raj() const;
    void        set_src_raj(double);
    int64_t     telescope_id() const;
    void        set_telescope_id(int64_t);
    double      tsamp() const;
    void        set_tsamp(double);
    double      tstart() const;
    void        set_tstart(double);

    int64_t data_type() const;
    void    set_data_type(int64_t);
    double  az_start() const;
    void    set_az_start(double);
    double  za_start() const;
    void    set_za_start(double);

    int64_t slow_time_bins() const;

    double tduration_secs() const;

  protected:
    std::map<int, std::shared_ptr<coarse_channel>> _coarse_channels;
    std::shared_ptr<h5_filterbank_file> _h5_file_handle=nullptr;

    // Read from h5 file
    double      _fch1;
    double      _foff;
    int64_t     _machine_id;
    int64_t     _nbits;
    int64_t     _nchans;
    int64_t     _nifs;
    std::string _source_name;
    double      _src_dej;
    double      _src_raj;
    int64_t     _telescope_id;
    double      _tsamp;
    double      _tstart;

    int64_t _data_type;
    double  _az_start;
    double  _za_start;

    // Derived values at read-time
    int64_t _num_coarse_channels;
    filterbank_channelization_revs _inferred_channelization;
    // slow time is number of spectra
    int64_t _slow_time_bins;
    double _tduration_secs;

  private:
};

} // namespace bliss
