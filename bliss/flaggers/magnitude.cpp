
#include <flaggers/magnitude.hpp>
#include <flaggers/flag_values.hpp>

#include <bland/ops.hpp>

using namespace bliss;

bland::ndarray bliss::flag_magnitude(const bland::ndarray &data, float threshold) {
    auto magnitude_mask = (data > threshold) * static_cast<uint8_t>(flag_values::magnitude);
    return magnitude_mask;
}

/**
 * return a mask with flags set for grid elements which have very large magnitudes
 * 
*/
filterbank_data bliss::flag_magnitude(filterbank_data fb_data, float threshold) {
    auto magnitude_mask = flag_magnitude(fb_data.data(), threshold);

    auto &mask = fb_data.mask();
    mask = mask + magnitude_mask;

    return fb_data;
}

filterbank_data bliss::flag_magnitude(filterbank_data fb_data) {
    auto data = fb_data.data();

    auto mean = bland::mean(data).scalarize<float>();
    auto stddev = bland::stddev(data).scalarize<float>();
    return flag_magnitude(fb_data, mean + 10*stddev);
}

observation_target bliss::flag_magnitude(observation_target observations, float threshold) {
    for (auto &filterbank : observations._filterbanks) {
        filterbank = flag_magnitude(filterbank, threshold);
    }
    return observations;
}

observation_target bliss::flag_magnitude(observation_target observations) {
    for (auto &filterbank : observations._filterbanks) {
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
