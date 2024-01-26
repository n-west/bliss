#pragma once

#include "noise_power.hpp"
#include <bland/bland.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace bliss {

struct h5_filterbank_file;

class filterbank_data {
  public:
    filterbank_data() = default;
    filterbank_data(h5_filterbank_file fb_file);
    filterbank_data(bland::ndarray data, bland::ndarray mask, double foff = 1);
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

    bland::ndarray &data();
    bland::ndarray &mask();
    // Set the mask to a new mask. A copy of underlying ndarray is not made
    // void            mask(const bland::ndarray &new_mask);

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
    // <KeysViewHDF5 ['data', 'mask']>
    // <HDF5 dataset "data": shape (16, 1, 1048576), type "<f4">
    // <HDF5 dataset "mask": shape (16, 1, 1048576), type "|u1">
    bland::ndarray _data;
    bland::ndarray _mask;

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
    /*
     * *DIMENSION_LABELS
     * *az_start
     * *data_type
     * *fch1
     * *foff
     * *machine_id
     * *nbits
     * *nchans
     * *nifs
     * *source_name
     * *src_dej
     * *src_raj
     * *telescope_id
     * *tsamp
     * *tstart
     * *za_start
     */

  private:
};

} // namespace bliss
