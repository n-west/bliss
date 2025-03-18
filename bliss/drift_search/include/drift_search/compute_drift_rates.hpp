
#include "core/frequency_drift_plane.hpp"
#include "core/integrate_drifts_options.hpp"

#include <vector>


namespace bliss {

    std::vector<frequency_drift_plane::drift_rate> compute_drifts(int time_steps, double foff, double tsamp, integrate_drifts_options options);

}