
#include "rfi/rfi.hpp"


using namespace bliss;


namespace detail {
    template <typename A, typename B>
    struct compare_threshold_op {
        static bool call(float x, float threshold) {
            return x>threshold;
        }
    };
} // namespace detail

bland::ndarray bliss::flag_hot_pixels(const bland::ndarray &spectrum_grid) {

    // Step 1: compute stats useful for thesholding individual pixels, we might want these passed in?
    auto stddev = bland::stddev(spectrum_grid);
    auto mean = bland::mean(spectrum_grid);
    /*
     * Impulsive RFI would show up clearly when the std/mean are taken across freq channels
       Constantish RFI would show up (but is harder due to smaller sample size)
    */

    // Step 2: Compute thresholds from those stats
    // 5 stddev is 99.999 942 of the population
    // We typically have how many spectra? expect this to show N on average per spectra
    // Should probably pick a factor to round to 0 samples for expected population size
    auto threshold = (mean + stddev * 5).scalarize<float>();


    // Step 3: Fill in the flag grid
    // for this it seems very useful to expose the striders already available, but I think we
    // also want to be able to access other values soon
    // TODO: define a lambda with the threshold, and pass that to dispatcher as a binary elemntwise operation

    auto spectrum_data = spectrum_grid.data_ptr<float>();
    
    bland::ndarray rfi_flags(spectrum_grid.shape(), bland::ndarray::datatype::uint8, spectrum_grid.device());

    return rfi_flags;
}