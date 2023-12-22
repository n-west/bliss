
#include <drift_search/local_maxima.hpp>

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <stdexcept>

using namespace bliss;

// Helper to abstract out increments
struct stride_helper {

    std::vector<int64_t> shape;
    std::vector<int64_t> stride;

    stride_helper(std::vector<int64_t> shape, std::vector<int64_t> stride) : shape(shape), stride(stride) {}

    int64_t to_linear_offset(const nd_coords &coords) const {
        // TODO compute linear index of curr_coords from shape and stride
        auto linear_offset = 0;
        for (int dim = 0; dim < shape.size(); ++dim) {
            linear_offset += coords[dim] * stride[dim]; // TODO: also add offset
        }
        return linear_offset;
    }
};

std::vector<component> bliss::find_local_maxima_above_threshold(doppler_spectrum &dedrifted_spectrum,
                                                                noise_stats       noise_stats,
                                                                float             snr_threshold,
                                                                std::vector<nd_coords> max_neighborhood) {
    // run through a max filter, what's the best way to establish neighborhood?

    auto hard_threshold = compute_signal_threshold(noise_stats, dedrifted_spectrum.integration_length(), snr_threshold);

    std::vector<component> maxima;

    // TODO: we also want to keep track of which flags contributed to *this* dedrifted spectrum so we can throw away
    // "hits" with too many (or the wrong kind) of flags
    auto &doppler_spectrum = dedrifted_spectrum.dedrifted_spectrum();
    if (doppler_spectrum.dtype() != bland::ndarray::datatype::float32) {
        throw std::runtime_error(
                "find_local_maxima_above_threshold: dedrifted doppler spectrum was not float. Only cpu "
                "float is supported right now");
    }
    auto          doppler_spectrum_data    = doppler_spectrum.data_ptr<float>();
    auto          doppler_spectrum_strides = doppler_spectrum.strides();
    stride_helper doppler_spectrum_strider(doppler_spectrum.shape(), doppler_spectrum_strides);

    // Use 1 to mark visited, then we can potentially replace this creation with a mask of above thresh to speed things
    // up a bit
    auto visited = bland::ndarray(dedrifted_spectrum.dedrifted_spectrum().shape(),
                                  1,
                                  bland::ndarray::datatype::uint8,
                                  bland::ndarray::dev::cpu);
    bland::fill(visited, 1);
    auto          visited_data    = visited.data_ptr<uint8_t>();
    auto          visited_shape   = visited.shape();
    auto          visited_strides = visited.strides();
    stride_helper visited_strider(visited_shape, visited_strides);
    nd_coords     curr_coord(visited_shape.size(), 0);

    auto numel = visited.numel();

    fmt::print("Looking through {} candidates with threshold {}\n", numel, hard_threshold);
    for (int64_t n = 0; n < numel; ++n) {
        // 1. Check if this is not visited & above threshold
        auto linear_visited_index          = visited_strider.to_linear_offset(curr_coord);
        auto linear_doppler_spectrum_index = doppler_spectrum_strider.to_linear_offset(curr_coord);
        if (visited_data[linear_visited_index] > 0) {
            // 2. Mark as visited
            visited_data[linear_visited_index] = 0; // We've been here now!

            // 3. Check that we're above our search threshold
            auto candidate_maxima_val = doppler_spectrum_data[linear_doppler_spectrum_index];
            if (candidate_maxima_val > hard_threshold) {

                // 4. Check if it is greater than surrounding neighborhood
                bool neighborhood_max = true;
                for (auto &neighbor_offset : max_neighborhood) {
                    bool in_bounds = true;
                    auto neighbor_coord = curr_coord;
                    for (int dim=0; dim < visited_shape.size(); ++dim) {
                        neighbor_coord[dim] += neighbor_offset[dim];
                        if (neighbor_coord[dim] < 0 || neighbor_coord[dim] >= visited_shape[dim]) {
                            in_bounds = false;
                            break;
                        }
                    }
                    
                    // check if the next coordinate is valid and not visited
                    if (in_bounds) {
                        auto linear_neighbor_index = doppler_spectrum_strider.to_linear_offset(neighbor_coord);
                        if (candidate_maxima_val > doppler_spectrum_data[linear_neighbor_index]) {
                            // we know this neighbor can't be a candidate maxima...
                            visited_data[visited_strider.to_linear_offset(neighbor_coord)] = 0;
                        } else {
                            neighborhood_max = false;
                            break;
                        }
                    }
                }

                // 3. Add to list of local maxima
                if (neighborhood_max) {
                    component c;
                    c.index_max       = curr_coord;
                    c.locations.push_back(curr_coord);
                    c.max_integration = candidate_maxima_val;
                    // TODO: this doesn't have the concept of a "component" the same way connected_components does... do
                    // we care?
                    maxima.push_back(c);
                }
            }
        }

        // 4. Increment ndindex
        // TODO: I decided this might be useful to add to the strider class...
        for (int dim = curr_coord.size() - 1; dim >= 0; --dim) {
            // If we're not at the end of this dim, keep going
            if (++curr_coord[dim] != visited_shape[dim]) {
                break;
            } else {
                // Otherwise, set it to 0 and move down to the next dim
                curr_coord[dim] = 0;
            }
        }
    }
    return maxima;
}
