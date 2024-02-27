#pragma once

#include <bland/bland.hpp>

#include "noise_power.hpp"
#include "integrate_drifts_options.hpp"
#include "frequency_drift_plane.hpp"
#include "hit.hpp"

#include <list>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>


namespace bliss {

struct coarse_channel {
    coarse_channel(bland::ndarray data,
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
    bland::ndarray data() const;
    bland::ndarray mask() const;
    void           set_mask(bland::ndarray new_mask);

    frequency_drift_plane integrated_drift_plane();
    void set_integrated_drift_plane(frequency_drift_plane integrated_plane);

    noise_stats noise_estimate() const;
    void        set_noise_estimate(noise_stats estimate);

    bool           has_hits();
    std::list<hit> hits() const;
    void           add_hits(std::list<hit> new_hits);

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

    int64_t        data_type() const;
    // void           data_type(int64_t);
    double         az_start() const;
    // void           az_start(double);
    double         za_start() const;
    // void           za_start(double);
    bland::ndarray _data;
    bland::ndarray _mask;
    // All values will be specific to the coarse channel
    // such as fch1 being the first channel of this coarse channel
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

    std::optional<noise_stats> _noise_stats;

    std::optional<frequency_drift_plane> _integrated_drift_plane;

    std::optional<bland::ndarray>           _dedrifted_spectrum;
    std::optional<integrated_flags>         _dedrifted_rfi;


    // TODO: I do not think we need to carry this around anymore since we'll track everything needed
    // in the frequency_drift_plane object
    std::optional<integrate_drifts_options> _drift_parameters;

    std::optional<std::list<hit>> _hits;
};

} // namespace bliss
