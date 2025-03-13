#pragma once

#include <bland/bland.hpp>

#include "frequency_drift_plane.hpp"
#include "hit.hpp"
#include "integrate_drifts_options.hpp"
#include "noise_power.hpp"

#include <functional> // std::function
#include <list>
#include <map>
#include <memory> // std::shared_ptr
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <variant> // std::variant

namespace bliss {

struct coarse_channel {

    coarse_channel(double      fch1,
                   double      foff,
                   int64_t     machine_id,
                   int64_t     nbits,
                   int64_t     nchans,
                   int64_t     ntsteps,
                   int64_t     nifs,
                   std::string source_name,
                   double      src_dej,
                   double      src_raj,
                   int64_t     telescope_id,
                   double      tsamp,
                   double      tstart,
                   int64_t     data_type,
                   double      az_start,
                   double      za_start,
                   int64_t        coarse_channel_number=0);

    coarse_channel(double                          fch1,
                   double                          foff,
                   std::optional<int64_t>          machine_id,
                   std::optional<int64_t>          nbits,
                   int64_t                         nchans,
                   int64_t                         ntsteps,
                   int64_t                         nifs,
                   std::string                     source_name,
                   std::optional<double>           src_dej,
                   std::optional<double>           src_raj,
                   std::optional<int64_t>          telescope_id,
                   double                          tsamp,
                   double                          tstart,
                   int64_t                         data_type,
                   std::optional<double>           az_start,
                   std::optional<double>           za_start,
                   int64_t        coarse_channel_number=0);

    // coarse_channel(std::function<bland::ndarray()> data,
    //                std::function<bland::ndarray()> mask,
    //                double                          fch1,
    //                double                          foff,
    //                int64_t                         machine_id,
    //                int64_t                         nbits,
    //                int64_t                         nchans,
    //                int64_t                         ntsteps,
    //                int64_t                         nifs,
    //                std::string                     source_name,
    //                double                          src_dej,
    //                double                          src_raj,
    //                int64_t                         telescope_id,
    //                double                          tsamp,
    //                double                          tstart,
    //                int64_t                         data_type,
    //                double                          az_start,
    //                double                          za_start,
    //                int64_t        coarse_channel_number=0);

    coarse_channel(std::function<bland::ndarray()> data,
                   std::function<bland::ndarray()> mask,
                   double                          fch1,
                   double                          foff,
                   std::optional<int64_t>          machine_id,
                   std::optional<int64_t>          nbits,
                   int64_t                         nchans,
                   int64_t                         ntsteps,
                   int64_t                         nifs,
                   std::string                     source_name,
                   std::optional<double>           src_dej,
                   std::optional<double>           src_raj,
                   std::optional<int64_t>          telescope_id,
                   double                          tsamp,
                   double                          tstart,
                   int64_t                         data_type,
                   std::optional<double>           az_start,
                   std::optional<double>           za_start,
                   int64_t        coarse_channel_number=0);

    coarse_channel(bland::ndarray data,
                   bland::ndarray mask,
                   double         fch1,
                   double         foff,
                   int64_t        machine_id,
                   int64_t        nbits,
                   int64_t        nchans,
                   int64_t        ntsteps,
                   int64_t        nifs,
                   std::string    source_name,
                   double         src_dej,
                   double         src_raj,
                   int64_t        telescope_id,
                   double         tsamp,
                   double         tstart,
                   int64_t        data_type,
                   double         az_start,
                   double         za_start,
                   int64_t        coarse_channel_number=0);

    bland::ndarray data();
    void                    set_data(bland::ndarray new_mask);

    bland::ndarray mask();
    void                    set_mask(bland::ndarray new_mask);

    std::optional<frequency_drift_plane> integrated_drift_plane();
    void                  set_integrated_drift_plane(frequency_drift_plane integrated_plane);

    noise_stats noise_estimate() const;
    void        set_noise_estimate(noise_stats estimate);

    bool           has_hits();
    std::list<hit> hits() const;
    void           set_hits(std::list<hit> new_hits);
    // void           set_hits(std::function<std::list<hit>()> find_hits_func);

    bland::ndarray::dev device();

    /**
     * Set the compute device for this coarse_channel. When `data` or `mask` is requested, it will first
     * be sent to this device if it is not already there.
     */
    void set_device(bland::ndarray::dev &device);
    void set_device(std::string_view device);

    void push_device();

    double fch1() const;
    // void        fch1(double);
    double foff() const;
    // void        foff(double);
    int64_t machine_id() const;
    // void        machine_id(int64_t);
    int64_t nbits() const;
    // void        nbits(int64_t);
    int64_t nchans() const;
    // void        nchans(int64_t);

    int64_t ntsteps() const;

    int64_t nifs() const;
    // void        nifs(int64_t);
    std::string source_name() const;
    // void        source_name(std::string);
    double src_dej() const;
    // void        src_dej(double);
    double src_raj() const;
    // void        src_raj(double);
    int64_t telescope_id() const;
    // void        telescope_id(int64_t);
    double tsamp() const;
    // void        tsamp(double);
    double tstart() const;
    // void        tstart(double);

    int64_t data_type() const;
    // void           data_type(int64_t);
    double az_start() const;
    // void           az_start(double);
    double za_start() const;
    // void           za_start(double);
    

    int64_t _coarse_channel_number; // the offset of this coarse channel within scan
    // All values will be specific to the coarse channel
    // such as fch1 being the first channel of this coarse channel
    double      _fch1;
    double      _foff;
    std::optional<int64_t>     _machine_id;
    std::optional<int64_t>     _nbits;
    int64_t     _nchans;
    int64_t     _nifs;
    std::string _source_name;
    std::optional<double>      _src_dej;
    std::optional<double>      _src_raj;
    std::optional<int64_t>     _telescope_id;
    double      _tsamp;
    double      _tstart;

    int64_t _ntsteps; // this x nchans is total data volume

    int64_t _data_type;
    std::optional<double>  _az_start;
    std::optional<double>  _za_start;

    bland::ndarray _data;
    bland::ndarray _mask;

    std::optional<noise_stats> _noise_stats;

    std::optional<frequency_drift_plane> _integrated_drift_plane;

    std::shared_ptr<std::list<hit>> _hits = nullptr;

    bland::ndarray::dev _device = bland::ndarray::dev::cpu;
};

} // namespace bliss
