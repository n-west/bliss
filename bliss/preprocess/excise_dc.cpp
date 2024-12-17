
#include <preprocess/excise_dc.hpp>

#include <bland/ndarray.hpp>
#include <bland/ops/ops.hpp>

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cmath>

using namespace bliss;

coarse_channel bliss::excise_dc(coarse_channel cc) {
    auto data = cc.data();
    auto dc_bin = data.size(1)/2;
    auto dc_data = bland::slice(data, 1, dc_bin, dc_bin+1, 1);
    dc_data = (bland::slice(data, 1, dc_bin-1, dc_bin, 1) + bland::slice(data, 1, dc_bin+1, dc_bin+2, 1))/2;
    cc.set_data(data);
    return cc;
}

scan bliss::excise_dc(scan sc) {
    sc.add_coarse_channel_transform([](coarse_channel cc) { return excise_dc(cc); });
    return sc;
}

observation_target bliss::excise_dc(observation_target ot) {
    for (auto &scan_data : ot._scans) {
        scan_data = excise_dc(scan_data);
    }
    return ot;
}

cadence bliss::excise_dc(cadence ca) {
    for (auto &target : ca._observations) {
        target = excise_dc(target);
    }
    return ca;
}
