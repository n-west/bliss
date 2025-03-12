
#include "core/coarse_channel.hpp"

#include <bland/config.hpp>

#include "fmt/format.h"
#include "fmt/ranges.h"

#include <variant>

using namespace bliss;

coarse_channel::coarse_channel(double      fch1,
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
                               int64_t        coarse_channel_number) :
        _fch1(fch1),
        _foff(foff),
        _machine_id(machine_id),
        _nbits(nbits),
        _nchans(nchans),
        _ntsteps(ntsteps),
        _nifs(nifs),
        _source_name(source_name),
        _src_dej(src_dej),
        _src_raj(src_raj),
        _telescope_id(telescope_id),
        _tsamp(tsamp),
        _tstart(tstart),
        _data_type(data_type),
        _az_start(az_start),
        _za_start(za_start),
        _coarse_channel_number(coarse_channel_number) {}

coarse_channel::coarse_channel(double              fch1,
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
                   int64_t                         coarse_channel_number) :
        _fch1(fch1),
        _foff(foff),
        _machine_id(machine_id),
        _nbits(nbits),
        _nchans(nchans),
        _ntsteps(ntsteps),
        _nifs(nifs),
        _source_name(source_name),
        _src_dej(src_dej),
        _src_raj(src_raj),
        _telescope_id(telescope_id),
        _tsamp(tsamp),
        _tstart(tstart),
        _data_type(data_type),
        _az_start(az_start),
        _za_start(za_start),
        _coarse_channel_number(coarse_channel_number) {}

// coarse_channel::coarse_channel(std::function<bland::ndarray()> data,
//                                std::function<bland::ndarray()> mask,
//                                double                          fch1,
//                                double                          foff,
//                                int64_t                         machine_id,
//                                int64_t                         nbits,
//                                int64_t                         nchans,
//                                int64_t                         ntsteps,
//                                int64_t                         nifs,
//                                std::string                     source_name,
//                                double                          src_dej,
//                                double                          src_raj,
//                                int64_t                         telescope_id,
//                                double                          tsamp,
//                                double                          tstart,
//                                int64_t                         data_type,
//                                double                          az_start,
//                                double                          za_start,
//                                int64_t                         coarse_channel_number) :
//         coarse_channel(fch1,
//                        foff,
//                        machine_id,
//                        nbits,
//                        nchans,
//                        ntsteps,
//                        nifs,
//                        source_name,
//                        src_dej,
//                        src_raj,
//                        telescope_id,
//                        tsamp,
//                        tstart,
//                        data_type,
//                        az_start,
//                        za_start,
//                        coarse_channel_number) {
//     _data = data();
//     _mask = mask();
// }

coarse_channel::coarse_channel(std::function<bland::ndarray()> data,
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
                                int64_t                         coarse_channel_number) :
        coarse_channel(fch1,
                       foff,
                       machine_id,
                       nbits,
                       nchans,
                       ntsteps,
                       nifs,
                       source_name,
                       src_dej,
                       src_raj,
                       telescope_id,
                       tsamp,
                       tstart,
                       data_type,
                       az_start,
                       za_start,
                       coarse_channel_number) {
    _data = data();
    _mask = mask();
}

coarse_channel::coarse_channel(bland::ndarray data,
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
                               int64_t        coarse_channel_number) :
        coarse_channel(fch1,
                       foff,
                       machine_id,
                       nbits,
                       nchans,
                       ntsteps,
                       nifs,
                       source_name,
                       src_dej,
                       src_raj,
                       telescope_id,
                       tsamp,
                       tstart,
                       data_type,
                       az_start,
                       za_start,
                       coarse_channel_number) {
    _data = data;
    _mask = mask;
}

bland::ndarray bliss::coarse_channel::data() {
    // TODO: add an option to compute graph to memoize the data read from disk
    // _data = _data.to(_device);
    return _data.to(_device);;
}

void bliss::coarse_channel::set_data(bland::ndarray new_data) {
    _data = new_data; // TODO: should we send _mask to _device?
}

bland::ndarray bliss::coarse_channel::mask() {
    // TODO: add an option to compute graph to memoize the mask read from disk
    // _mask = _mask.to(_device);
    return _mask.to(_device);
}

void bliss::coarse_channel::set_mask(bland::ndarray new_mask) {
    _mask = new_mask; // TODO: should we send _mask to _device?
}

noise_stats bliss::coarse_channel::noise_estimate() const {
    if (_noise_stats.has_value()) {
        return _noise_stats.value();
    } else {
        fmt::print("coarse_channel::noise_estimate: requested noise estimate which does not exist yet.");
        throw std::logic_error("coarse_channel::noise_estimate: requested noise estimate which does not exist");
    }
}

void bliss::coarse_channel::set_noise_estimate(noise_stats estimate) {
    _noise_stats = estimate;
}

bool bliss::coarse_channel::has_hits() {
    if (_hits == nullptr) {
        return false;
    } else {
        return true;
    }
}

std::list<hit> bliss::coarse_channel::hits() const {
    if (_hits == nullptr) {
        throw std::logic_error("hits not set");
    }
    if (std::holds_alternative<std::function<std::list<hit>()>>(*_hits)) {
        *_hits = std::get<std::function<std::list<hit>()>>(*_hits)();
    }
    auto requested_hits = std::get<std::list<hit>>(*_hits);
    return requested_hits;
}

void bliss::coarse_channel::set_hits(std::list<hit> new_hits) {
    _hits = std::make_shared<std::variant<std::list<hit>, std::function<std::list<hit>()>>>(new_hits);
}

void bliss::coarse_channel::set_hits(std::function<std::list<hit>()> find_hits_func) {
    _hits = std::make_shared<std::variant<std::list<hit>, std::function<std::list<hit>()>>>(find_hits_func);
}

bland::ndarray::dev bliss::coarse_channel::device() {
    return _device;
}

void bliss::coarse_channel::set_device(bland::ndarray::dev &device) {
    if (device.device_type == bland::ndarray::dev::cuda.device_type ||
        device.device_type == bland::ndarray::dev::cuda_managed.device_type) {
        if (!bland::g_config.check_is_valid_cuda_device(device.device_id)) {
            fmt::print("The selected device id either does not exist or has a compute capability that is not compatible with this build\n");
            throw std::runtime_error("set_device received invalid cuda device");
        }
    }
    _device = device;
}

void bliss::coarse_channel::set_device(std::string_view device) {
    // convert to dev and defer to other set_device which does checking
    bland::ndarray::dev proper_dev = device;
    set_device(proper_dev);
}

void bliss::coarse_channel::push_device() {
    _mask = _mask.to(_device);
    _data = _data.to(_device);
    if (_integrated_drift_plane != nullptr && std::holds_alternative<frequency_drift_plane>(*_integrated_drift_plane)) {
        auto &idp = std::get<frequency_drift_plane>(*_integrated_drift_plane);
        idp.set_device(_device);
        idp.push_device();
    }
}

double bliss::coarse_channel::fch1() const {
    return _fch1;
}
// void bliss::scan::fch1(double fch1) {
//     _fch1 = fch1;
// }

double bliss::coarse_channel::foff() const {
    return _foff;
}
// void bliss::scan::foff(double foff) {
//     _foff = foff;
// }

int64_t bliss::coarse_channel::machine_id() const {
    return _machine_id.value();
}
// void bliss::scan::machine_id(int64_t machine_id) {
//     _machine_id = machine_id;
// }

int64_t bliss::coarse_channel::nbits() const {
    return _nbits.value();
}
// void bliss::scan::nbits(int64_t nbits) {
//     _nbits = nbits;
// }

int64_t bliss::coarse_channel::nchans() const {
    return _nchans;
}
// void bliss::scan::nchans(int64_t nchans) {
//     _nchans = nchans;
// }

int64_t bliss::coarse_channel::ntsteps() const {
    return _ntsteps;
}
// void bliss::scan::ntsteps(int64_t ntsteps) {
//     _ntsteps = ntsteps;
// }

int64_t bliss::coarse_channel::nifs() const {
    return _nifs;
}
// void bliss::scan::nifs(int64_t nifs) {
//     _nifs = nifs;
// }

std::string bliss::coarse_channel::source_name() const {
    return _source_name;
}
// void bliss::scan::source_name(std::string source_name) {
//     _source_name = source_name;
// }

double bliss::coarse_channel::src_dej() const {
    return _src_dej.value();
}
// void bliss::scan::src_dej(double src_dej) {
//     _src_dej = src_dej;
// }

double bliss::coarse_channel::src_raj() const {
    return _src_raj.value();
}
// void bliss::scan::src_raj(double src_raj) {
//     _src_raj = src_raj;
// }

int64_t bliss::coarse_channel::telescope_id() const {
    return _telescope_id.value();
}
// void bliss::scan::telescope_id(int64_t telescope_id) {
//     _telescope_id = telescope_id;
// }

double bliss::coarse_channel::tsamp() const {
    return _tsamp;
}
// void bliss::scan::tsamp(double tsamp) {
//     _tsamp = tsamp;
// }

double bliss::coarse_channel::tstart() const {
    return _tstart;
}
// void bliss::scan::tstart(double tstart) {
//     _tstart = tstart;
// }

int64_t bliss::coarse_channel::data_type() const {
    return _data_type;
}
// void bliss::scan::data_type(int64_t data_type) {
//     _data_type = data_type;
// }

double bliss::coarse_channel::az_start() const {
    return _az_start.value();
}
// void bliss::scan::az_start(double az_start) {
//     _az_start = az_start;
// }

double bliss::coarse_channel::za_start() const {
    return _za_start.value();
}
// void bliss::scan::za_start(double za_start) {
//     _za_start = za_start;
// }

frequency_drift_plane bliss::coarse_channel::integrated_drift_plane() {
    if (_integrated_drift_plane == nullptr) {
        throw std::runtime_error("integrated_drift_plane not set");
    }
    // TODO: this used to hold on to a memoized copy of the result, but is set up to not do that now
    // in order to save VRAM at a cost of recomputing as needed. This could be passed in as a compute
    // graph option
    if (std::holds_alternative<std::function<frequency_drift_plane()>>(*_integrated_drift_plane)) {
        auto integrated_drift_plane = std::get<std::function<frequency_drift_plane()>>(*_integrated_drift_plane)();

        integrated_drift_plane.set_device(_device);
        return integrated_drift_plane;
    } else {
        auto &ddp = std::get<frequency_drift_plane>(*_integrated_drift_plane);
        ddp.set_device(_device);
        return ddp;
    }
}

void bliss::coarse_channel::set_integrated_drift_plane(frequency_drift_plane integrated_plane) {
    _integrated_drift_plane =
            std::make_shared<std::variant<frequency_drift_plane, std::function<frequency_drift_plane()>>>(
                    integrated_plane);
}

void bliss::coarse_channel::set_integrated_drift_plane(
        std::function<frequency_drift_plane()> integrated_plane_generator) {
    _integrated_drift_plane =
            std::make_shared<std::variant<frequency_drift_plane, std::function<frequency_drift_plane()>>>(
                    integrated_plane_generator);
}
