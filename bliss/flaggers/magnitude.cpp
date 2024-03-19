
#include <core/flag_values.hpp>
#include <flaggers/magnitude.hpp>

#include <bland/ops/ops.hpp>

using namespace bliss;

bland::ndarray bliss::flag_magnitude(const bland::ndarray &data, float threshold) {
    auto magnitude_mask = (data > threshold) * static_cast<uint8_t>(flag_values::magnitude);
    return magnitude_mask;
}


/**
 * return a mask with flags set for grid elements which have very large magnitudes
 * 
*/
coarse_channel bliss::flag_magnitude(coarse_channel cc_data, float threshold) {
    auto magnitude_mask = flag_magnitude(cc_data.data(), threshold);

    bland::ndarray mask = cc_data.mask();
    mask = mask + magnitude_mask;
    cc_data.set_mask(mask);
    return cc_data;
}

coarse_channel bliss::flag_magnitude(coarse_channel cc_data) {
    auto data = cc_data.data();

    auto mean = bland::mean(data).scalarize<float>();
    auto stddev = bland::stddev(data).scalarize<float>();
    return flag_magnitude(cc_data, mean + 10*stddev);
}

/**
 * return a mask with flags set for grid elements which have very large magnitudes
 * 
*/
scan bliss::flag_magnitude(scan fil_data, float threshold) {
    auto number_coarse_channels = fil_data.get_number_coarse_channels();
    for (auto cc_index = 0; cc_index < number_coarse_channels; ++cc_index) {
        auto cc = fil_data.get_coarse_channel(cc_index);
        *cc = flag_magnitude(*cc, threshold);
    }
    return fil_data;
}

scan bliss::flag_magnitude(scan fil_data) {
    auto number_coarse_channels = fil_data.get_number_coarse_channels();
    for (auto cc_index = 0; cc_index < number_coarse_channels; ++cc_index) {
        auto cc = fil_data.get_coarse_channel(cc_index);
        *cc = flag_magnitude(*cc);
    }
    return fil_data;
}

observation_target bliss::flag_magnitude(observation_target observations, float threshold) {
    for (auto &filterbank : observations._scans) {
        filterbank = flag_magnitude(filterbank, threshold);
    }
    return observations;
}

observation_target bliss::flag_magnitude(observation_target observations) {
    for (auto &filterbank : observations._scans) {
        filterbank = flag_magnitude(filterbank);
    }
    return observations;
}

cadence bliss::flag_magnitude(cadence observations, float threshold) {
    for (auto &observation : observations._observations) {
        observation = flag_magnitude(observation, threshold);
    }
    return observations;
}

cadence bliss::flag_magnitude(cadence observations) {
    for (auto &observation : observations._observations) {
        observation = flag_magnitude(observation);
    }
    return observations;
}
