#pragma once

#include "coarse_channel.hpp"

#include <bland/bland.hpp>

#include <cstdint>
#include <map>
#include <string>
#include <string_view>

namespace bliss {

struct h5_filterbank_file;

class scan {
  public:
    scan() = default;

    scan(std::map<int, std::shared_ptr<coarse_channel>> coarse_channels);

    /**
     * new scan backed by the given `h5_filterbank_file`
     */
    scan(h5_filterbank_file fb_file, int num_fine_channels_per_coarse=0);

    /**
     * new scan backed by the filterbank file at `file_path`
     */
    scan(std::string_view file_path, int num_fine_channels_per_coarse=0);

    /**
     * read the coarse channel at given index and return shared ownership of it.
     *
     * The `coarse_channel` has similar metadata to a `scan` object
     * but is specific to the `coarse_channel`, for example `fch1` and `nchan`
     * represent the frequency of the first fine channel in this coarse channel
     * and the number of fine channels in this channel.
     *
     * It might exist in an internal cache in which case it's simply returned without
     * disk access. The cache may remove a `coarse_channel` without notice so
     * ownership of `coarse_channels` is shared.
     */
    std::shared_ptr<coarse_channel> read_coarse_channel(int coarse_channel_index = 0);

    /**
     * Similar to `get_coarse_channel` but returns nullptr if the coarse_channel isn't
     * already in memory (has not been read from disk / used yet). This is useful for
     * gathering results / operating on all coarse channels that are already loaded
    */
    std::shared_ptr<coarse_channel> peak_coarse_channel(int coarse_channel_index = 0);

    /**
     * A function that will be called on each coarse_channel when read with read_coarse_channel
     * or poke_coarse_channel
     */
    void add_coarse_channel_transform(std::function<coarse_channel(coarse_channel)> transform);

    /**
     * return the coarse channel index that the given frequency is in
     * 
     * useful for reinvestigating hits by looking up frequency
     * 
     * In the future this may change name or return the actual coarse channel
    */
    int get_coarse_channel_with_frequency(double frequency) const;

    /**
     * get the number of coarse channels in this filterbank
     *
     * This value is derived from known channelizations of BL backends and
     * the actual number of fine channels in this filterbank
     */
    int get_number_coarse_channels() const;

    std::string get_file_path() const;

    /**
     * gather hits in all coarse channels of this scan and return as a single list
     */
    std::list<hit> hits();

    std::pair<float, float> get_drift_range();

    bland::ndarray::dev device();
    void set_device(bland::ndarray::dev &device, bool verbose=true);
    void set_device(std::string_view device, bool verbose=true);
    void push_device();

    /**
     * create a new scan consisting of the selected coarse channel
     */
    scan slice_scan_channels(int64_t start_channel = 0, int64_t count = 1);

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

    int64_t ntsteps() const;

    double tduration_secs() const;

  protected:
    // TODO (design note): This shared_ptr is shared between all slices which means set_device
    // on higher layers or slices of this scan will effect every slice (or even the global) which
    // may inadverdently increase vmem. It's a bit more work and not worth it right now because
    // it's not an active issue, but we can also keep track of "active_coarse_channels" which are
    // only those within the current slice / copy and only have set_device, etc effect those channels
    // std::string _original_file_path={};
    std::map<int, std::shared_ptr<coarse_channel>> _coarse_channels;
    std::shared_ptr<h5_filterbank_file>            _h5_file_handle = nullptr;
    std::vector<std::function<coarse_channel(coarse_channel)>> _coarse_channel_pipeline;

    // Read from h5 file
    double      _fch1;
    double      _foff;
    std::optional<int64_t>     _machine_id;
    int64_t     _nbits;
    int64_t     _nchans;
    int64_t     _nifs;
    std::string _source_name;
    std::optional<double>      _src_dej;
    std::optional<double>      _src_raj;
    std::optional<int64_t>     _telescope_id;
    double      _tsamp;
    double      _tstart;

    int64_t _data_type;
    std::optional<double>  _az_start;
    std::optional<double>  _za_start;


    // Derived OR inferred
    int _fine_channels_per_coarse;
    // Derived values at read-time
    int64_t _num_coarse_channels;
    int64_t _coarse_channel_offset = 0;

    // slow time is number of spectra
    int64_t _ntsteps;
    double  _tduration_secs;

    bland::ndarray::dev _device = bland::ndarray::dev::cpu;

  private:
};

} // namespace bliss
