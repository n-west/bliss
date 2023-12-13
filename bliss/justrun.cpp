
#include <estimators/noise_estimate.hpp>
#include <file_types/h5_filterbank_file.hpp>
#include <file_types/filterbank_data.hpp>
#include <flaggers/spectral_kurtosis.hpp>
#include <flaggers/magnitude.hpp>
#include <spectrumsum/hit_search.hpp>
#include <spectrumsum/spectrumsum.hpp>
#include "fmt/core.h"
#include <fmt/ranges.h>

#include <H5Cpp.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int main()
{

  auto fil_data = bliss::filterbank_data(
      "/home/nathan/datasets/voyager_2020_data/single_coarse_guppi_59046_80354_DIAG_VOYAGER-1_0012.rawspec.0000.h5");

  // TODO: add support for vector of looks (a cadence dtype that holds all dwells)

  auto flagged_fil = bliss::flag_spectral_kurtosis(fil_data, 0.01f, 10.0f); // SK > 10 && SK < .01
  // auto flagged_fil = bliss::flag_magnitude(flagged_fil, 10.0f); // mag > mean + std*10
  // auto flagged_fil = bliss::flag_filter_rolloff(flagged_fil, .05); // % of band edges
  // auto flagged_fil = bliss::flag_ood(flagged_fil, .01); // < 1% change belonging to predicted noise only distribution

  // auto flagged_fil = bliss::noise_power_estimate(fil_data, 0.01f, 10.0f); // estimate noise power of unflagged data

  // auto dedrifted_fil = bliss::spectrum_sum(flagged_fil); // integrate along drift lines

  // auto hits = bliss::hit_search(dedrifted_fil)

  // bliss::write_hits(hits);

  std::cout << std::endl;
}
