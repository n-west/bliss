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

class scan {
  public:
    scan() = default;

    /**
     * new scan backed by the given `h5_filterbank_file`
     */
    scan(h5_filterbank_file fb_file);

    /**
     * new scan backed by the filterbank file at `file_path`
     */
    scan(std::string_view file_path);

    scan(bland::ndarray data,
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

    scan(double      fch1,
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

    /**
     * filter_channelization_revs describes the channelization configuration known to
     * be used in BL backends and data. The tuple order is
     * * number of fine channels per coarse channel
     * * frequency resolution (equivalent to foff filterbank md and inverse of Fs)
     * * time resolution (equivalent to tsamp filterbank md)
     * * name of revision from Lebofsky et al
     *
     * The best paper reference for this information is
     * "The Breakthrough Listen Search for Intelligent Life: Public Data, Formats, Reduction and Archiving"
     * available @ https://arxiv.org/abs/1906.07391
     *
     * This may eventually require it's own POD class
     */
    using filterbank_channelization_revs = std::tuple<int, double, double, const char *>;

    /**
     * read the coarse channel at given index and return shared ownership of it.
     *
     * The `coarse_channel` has similar metadata to a `scan` object
     * but is specific to the `corase_channel`, for example `fch1` and `nchan`
     * represent the frequency of the first fine channel in this coarse channel
     * and the number of fine channels in this channel.
     *
     * It might exist in an internal cache in which case it's simply returned without
     * disk access. The cache may remove a `coarse_channel` without notice so
     * ownership of `coarse_channels` is shared.
     */
    std::shared_ptr<coarse_channel> get_coarse_channel(int coarse_channel_index = 0);

    filterbank_channelization_revs get_channelization();

    /**
     * return the coarse channel index that the given frequency is in
     * 
     * useful for reinvestigating hits by looking up frequency
    */
    int get_coarse_channel_with_frequency(double frequency);

    /**
     * get the number of coarse channels in this filterbank
     *
     * This value is derived from known channelizations of BL backends and
     * the actual number of fine channels in this filterbank
     */
    int get_number_coarse_channels();

    /**
     * gather hits in all coarse channels of this scan and return as a single list
     */
    std::list<hit> hits();

    /**
     * create a new scan consisting of the selected coarse channel
     */
    scan slice_scan_channels(int start_channel = 0, int count = 1);

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
    std::shared_ptr<h5_filterbank_file>            _h5_file_handle = nullptr;

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
    int64_t _coarse_channel_offset = 0;

    filterbank_channelization_revs _inferred_channelization;
    // slow time is number of spectra
    int64_t _slow_time_bins;
    double  _tduration_secs;

  private:
};

} // namespace bliss
