
#include <preprocess/normalize.hpp>

#include <bland/ndarray.hpp>
#include <bland/ndarray_slice.hpp>
#include <bland/ops/ops.hpp>

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cmath>

using namespace bliss;

coarse_channel bliss::normalize(coarse_channel cc) {
    // data is (or can be) a deferred tensor
    // we might need to make a copy of cc rather than pass a ref since we're accessing the deferred array and setting it all in one go
    auto cc_ptr = std::make_shared<coarse_channel>(cc);
    cc.set_data(bland::ndarray_deferred([cc_data=cc_ptr]() {
        auto norm_value = bland::max(cc_data->data());
        return bland::divide(cc_data->data(), norm_value);
    }));
    // cc.set_data(bland::divide(cc.data(), h));
    return cc;
}

scan bliss::normalize(scan sc) {
    auto number_coarse_channels = sc.get_number_coarse_channels();
    for (auto cc_index = 0; cc_index < number_coarse_channels; ++cc_index) {
        auto cc = sc.read_coarse_channel(cc_index);
        *cc = normalize(*cc);
    }
    return sc;
}

observation_target bliss::normalize(observation_target ot) {
    for (auto &scan_data : ot._scans) {
        scan_data = normalize(scan_data);
    }
    return ot;
}

cadence bliss::normalize(cadence ca) {
    for (auto &target : ca._observations) {
        target = normalize(target);
    }
    return ca;
}
