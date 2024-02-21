#pragma once

#include <bland/bland.hpp>

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>

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

    bland::ndarray &data(int coarse_channel=0);
    bland::ndarray &mask(int coarse_channel=0);

    /**
     * get the number of coarse channels in this filterbank
     * 
     * This value is derived from known channelizations of BL backends and
     * the actual number of fine channels in this filterbank
    */
    int get_number_coarse_channels();
    using filterbank_channelization_revs = std::tuple<int, double, double, const char*>;

    // Setters and getters for values read from disk
    double      fch1();
    void        fch1(double);
    double      foff();
    void        foff(double);
    int64_t     machine_id();
    void        machine_id(int64_t);
    int64_t     nbits();
    void        nbits(int64_t);
    int64_t     nchans();
    void        nchans(int64_t);
    int64_t     nifs();
    void        nifs(int64_t);
    std::string source_name();
    void        source_name(std::string);
    double      src_dej();
    void        src_dej(double);
    double      src_raj();
    void        src_raj(double);
    int64_t     telescope_id();
    void        telescope_id(int64_t);
    double      tsamp();
    void        tsamp(double);
    double      tstart();
    void        tstart(double);

    int64_t data_type();
    void    data_type(int64_t);
    double  az_start();
    void    az_start(double);
    double  za_start();
    void    za_start(double);

  protected:
    std::map<int, bland::ndarray> _data;
    std::map<int, bland::ndarray> _mask;
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

  private:
};

} // namespace bliss
