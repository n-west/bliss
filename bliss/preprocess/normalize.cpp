
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
    auto data = cc.data();
    auto norm_value = bland::max(data);
    auto normalized = bland::divide(data, norm_value);
    cc.set_data(normalized);
    return cc;
}

scan bliss::normalize(scan sc) {
    sc.add_coarse_channel_transform([](coarse_channel cc) { return normalize(cc); });
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
