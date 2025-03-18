
#include <drift_search/compute_drift_rates.hpp>

using namespace bliss;

std::vector<frequency_drift_plane::drift_rate> bliss::compute_drifts(int time_steps, double foff, double tsamp, integrate_drifts_options options) {
    auto maximum_drift_time_span = time_steps - 1;

    // Convert the drift options to specific drift info
    auto foff_Hz = (foff)*1e6;
    double unit_drift_resolution = foff_Hz/((time_steps-1) * tsamp);
    // Round the drift bounds to multiple of unit_drift_resolution
    auto search_resolution_Hz_sec = unit_drift_resolution * options.resolution;
    auto rounded_low_drift_Hz_sec = std::round(options.low_rate_Hz_per_sec / unit_drift_resolution) * unit_drift_resolution;
    auto rounded_high_drift_Hz_sec = std::round(options.high_rate_Hz_per_sec / unit_drift_resolution) * unit_drift_resolution;

    int64_t number_drifts = std::abs((rounded_high_drift_Hz_sec - rounded_low_drift_Hz_sec) / (search_resolution_Hz_sec));

    fmt::print("INFO: Searching {} drift rates from {} Hz/sec to {} Hz/sec in increments of {} Hz/sec\n", number_drifts, rounded_low_drift_Hz_sec, rounded_high_drift_Hz_sec, std::abs(search_resolution_Hz_sec));
    if (options.round_to_multiple_of_data && (number_drifts % time_steps != 0)) {
        auto rounded_number_drifts = time_steps * (1 + number_drifts / time_steps);
        auto extra_drifts = rounded_number_drifts - number_drifts;
        auto extra_drifts_on_low = extra_drifts / 2;
        auto extra_drifts_on_high = extra_drifts - extra_drifts_on_low;
        auto extra_Hz_on_low = extra_drifts_on_low * std::abs(search_resolution_Hz_sec);
        auto extra_Hz_on_high = extra_drifts_on_high * std::abs(search_resolution_Hz_sec);
        rounded_low_drift_Hz_sec -= extra_Hz_on_low;
        rounded_high_drift_Hz_sec += extra_Hz_on_high;
        number_drifts = rounded_number_drifts;
        fmt::print("INFO: Rounding drift rates to multiple of data gives updated drift range {} Hz/sec to {} Hz/sec\n", rounded_low_drift_Hz_sec, rounded_high_drift_Hz_sec);
    }
    std::vector<frequency_drift_plane::drift_rate> drift_rate_info;
    drift_rate_info.reserve(number_drifts);
    for (int index = 0; index < number_drifts; ++index) {
        auto drift_rate = rounded_low_drift_Hz_sec + index * std::abs(search_resolution_Hz_sec);
        frequency_drift_plane::drift_rate rate;
        rate.index_in_plane = index;
        // You'll get automatic truncation without intentional rounding
        rate.drift_channels_span = std::lround(drift_rate * ((maximum_drift_time_span)*tsamp)/(foff_Hz));

        // The actual slope of that drift (number channels / number time steps)
        // Should be equivalent to a unitless drift rate (drift_rate * tsamp/foff)
        auto m = static_cast<float>(rate.drift_channels_span) / static_cast<float>(maximum_drift_time_span);

        rate.drift_rate_slope = m;
        rate.drift_rate_Hz_per_sec = drift_rate;
        // If a single time step crosses more than 1 channel, there is smearing over multiple channels
        auto smeared_channels = std::round(std::abs(m));

        int desmear_binwidth = 1;
        if (options.desmear) {
            desmear_binwidth = std::max(1.0F, smeared_channels);
        }
        rate.desmeared_bins = desmear_binwidth;

        drift_rate_info.push_back(rate);
    }
    return drift_rate_info;
}
