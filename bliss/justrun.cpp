
#include <estimators/noise_estimate.hpp>
#include <file_types/h5_filterbank_file.hpp>
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

  std::cout << "Hello, this is bliss-- the Breakthrough Listen Interesting Signal Search" << std::endl;

  auto fil_file = bliss::h5_filterbank_file(
      "/home/nathan/datasets/voyager_2020_data/single_coarse_guppi_59046_80354_DIAG_VOYAGER-1_0012.rawspec.0000.h5");

  std::cout << "class (new method): " << fil_file.read_file_attr<std::string>("CLASS") << std::endl;

  auto spectrum_grid = fil_file.read_data();

  auto tsamp  = fil_file.read_data_attr<double>("tsamp");
  auto nsteps = spectrum_grid.size(0);

  assert(spectrum_grid.size(1) == fil_file.read_data_attr<int64_t>("nchans"));
  auto nchans = spectrum_grid.size(1);
  auto foff   = fil_file.read_data_attr<double>("foff") * 1e6;
  std::cout << nchans << " channels representing a total of " << (foff * nchans) / 1e6 << " MHz" << std::endl;

  std::cout << "Samples represent time of " << tsamp << std::endl;
  // Convince myself of the -1
  auto drift_resolution = foff / ((nsteps - 1) * tsamp);
  std::cout << nsteps << " represent " << tsamp / nsteps << " sec each"
            << " giving a minimum drift search resolution of " << drift_resolution << " Hz" << std::endl;

  std::cout << "Some notable values:" << std::endl;
  fmt::print("{:8f}\n", spectrum_grid.data_ptr<float>()[0 + spectrum_grid.strides()[0] * 0]);
  fmt::print("{:8f}\n", spectrum_grid.data_ptr<float>()[0 + spectrum_grid.strides()[0] * 1]);
  fmt::print("{:8f}\n", spectrum_grid.data_ptr<float>()[0 + spectrum_grid.strides()[0] * 2]);
  fmt::print("{:8f}\n", spectrum_grid.data_ptr<float>()[0 + spectrum_grid.strides()[0] * 3]);
  fmt::print("{:8f}\n", spectrum_grid.data_ptr<float>()[0 + spectrum_grid.strides()[0] * 4]);
  fmt::print("{:8f}\n", spectrum_grid.data_ptr<float>()[0 + spectrum_grid.strides()[0] * 5]);
  fmt::print("{:8f}\n", spectrum_grid.data_ptr<float>()[0 + spectrum_grid.strides()[0] * 6]);
  fmt::print("{:8f}\n", spectrum_grid.data_ptr<float>()[0 + spectrum_grid.strides()[0] * 7]);
  fmt::print("{:8f}\n", spectrum_grid.data_ptr<float>()[0 + spectrum_grid.strides()[0] * 8]);
  fmt::print("{:8f}\n", spectrum_grid.data_ptr<float>()[0 + spectrum_grid.strides()[0] * 9]);
  fmt::print("{:8f}\n", spectrum_grid.data_ptr<float>()[0 + spectrum_grid.strides()[0] * 10]);
  fmt::print("{:8f}\n", spectrum_grid.data_ptr<float>()[0 + spectrum_grid.strides()[0] * 11]);
  fmt::print("{:8f}\n", spectrum_grid.data_ptr<float>()[0 + spectrum_grid.strides()[0] * 12]);
  fmt::print("{:8f}\n", spectrum_grid.data_ptr<float>()[0 + spectrum_grid.strides()[0] * 13]);
  fmt::print("{:8f}\n", spectrum_grid.data_ptr<float>()[0 + spectrum_grid.strides()[0] * 14]);
  fmt::print("{:8f}\n", spectrum_grid.data_ptr<float>()[0 + spectrum_grid.strides()[0] * 15]);

  auto mean_and_var = bliss::noise_power_estimate(spectrum_grid, bliss::noise_power_estimator::TURBO_SETI);
  fmt::print("Expected stats of spectrum: {}   {}\n", mean_and_var.mean(), mean_and_var.var());

  std::cout << "Some notable values:" << std::endl;
  fmt::print("{:8f}\n", spectrum_grid.data_ptr<float>()[0 + spectrum_grid.strides()[0] * 0]);
  fmt::print("{:8f}\n", spectrum_grid.data_ptr<float>()[0 + spectrum_grid.strides()[0] * 1]);
  fmt::print("{:8f}\n", spectrum_grid.data_ptr<float>()[0 + spectrum_grid.strides()[0] * 2]);
  fmt::print("{:8f}\n", spectrum_grid.data_ptr<float>()[1 + spectrum_grid.strides()[0] * 3]);
  fmt::print("{:8f}\n", spectrum_grid.data_ptr<float>()[1 + spectrum_grid.strides()[0] * 4]);
  fmt::print("{:8f}\n", spectrum_grid.data_ptr<float>()[1 + spectrum_grid.strides()[0] * 5]);

//   // There are 16 sums per drift estimate
// //   mean_and_std.mean *= 16;
// //   mean_and_std.var = std::sqrt(16) * mean_and_std.var;

//   std::cout << "Expected stats of drift spectrum: " << mean_and_var.mean() << " " << mean_and_var.var() << std::endl;
  auto drift_spectrum = bliss::spectrum_sum(spectrum_grid);

  fmt::print("The shape of drift spectrum is {} with strides {}\n", drift_spectrum.shape(), drift_spectrum.strides());
  std::cout << "Drifts starting at freq 0" << std::endl;
  fmt::print("{:8f}\n", drift_spectrum.data_ptr<float>()[drift_spectrum.strides()[0] * 0] );
  fmt::print("{:8f}\n", drift_spectrum.data_ptr<float>()[drift_spectrum.strides()[0] * 1] );
  fmt::print("{:8f}\n", drift_spectrum.data_ptr<float>()[drift_spectrum.strides()[0] * 2] );
  fmt::print("{:8f}\n", drift_spectrum.data_ptr<float>()[drift_spectrum.strides()[0] * 3] );
  fmt::print("{:8f}\n", drift_spectrum.data_ptr<float>()[drift_spectrum.strides()[0] * 4] );
  fmt::print("{:8f}\n", drift_spectrum.data_ptr<float>()[drift_spectrum.strides()[0] * 5] );
  fmt::print("{:8f}\n", drift_spectrum.data_ptr<float>()[drift_spectrum.strides()[0] * 6] );
  fmt::print("{:8f}\n", drift_spectrum.data_ptr<float>()[drift_spectrum.strides()[0] * 7] );
  fmt::print("{:8f}\n", drift_spectrum.data_ptr<float>()[drift_spectrum.strides()[0] * 8] );
  fmt::print("{:8f}\n", drift_spectrum.data_ptr<float>()[drift_spectrum.strides()[0] * 9] );
  fmt::print("{:8f}\n", drift_spectrum.data_ptr<float>()[drift_spectrum.strides()[0] * 10]);
  fmt::print("{:8f}\n", drift_spectrum.data_ptr<float>()[drift_spectrum.strides()[0] * 11]);
  fmt::print("{:8f}\n", drift_spectrum.data_ptr<float>()[drift_spectrum.strides()[0] * 12]);
  fmt::print("{:8f}\n", drift_spectrum.data_ptr<float>()[drift_spectrum.strides()[0] * 13]);
  fmt::print("{:8f}\n", drift_spectrum.data_ptr<float>()[drift_spectrum.strides()[0] * 14]);
  fmt::print("{:8f}\n", drift_spectrum.data_ptr<float>()[drift_spectrum.strides()[0] * 15]);

  std::cout << "Drifts starting at freq 1" << std::endl;
  std::cout << drift_spectrum.data_ptr<float>()[1 + drift_spectrum.strides()[0] * 0] << std::endl;
  std::cout << drift_spectrum.data_ptr<float>()[1 + drift_spectrum.strides()[0] * 1] << std::endl;
  std::cout << drift_spectrum.data_ptr<float>()[1 + drift_spectrum.strides()[0] * 2] << std::endl;
  std::cout << drift_spectrum.data_ptr<float>()[1 + drift_spectrum.strides()[0] * 3] << std::endl;
  std::cout << drift_spectrum.data_ptr<float>()[1 + drift_spectrum.strides()[0] * 4] << std::endl;

  std::cout << "Drifts starting at freq 2" << std::endl;
  std::cout << drift_spectrum.data_ptr<float>()[2 + drift_spectrum.strides()[0] * 0] << std::endl;
  std::cout << drift_spectrum.data_ptr<float>()[2 + drift_spectrum.strides()[0] * 1] << std::endl;
  std::cout << drift_spectrum.data_ptr<float>()[2 + drift_spectrum.strides()[0] * 2] << std::endl;
  std::cout << drift_spectrum.data_ptr<float>()[2 + drift_spectrum.strides()[0] * 3] << std::endl;
  std::cout << drift_spectrum.data_ptr<float>()[2 + drift_spectrum.strides()[0] * 4] << std::endl;


//   bliss::hitsearch(drift_spectrum, mean_and_var, 10);

//   /**
//    * eigen (standard, should be ok for gpu, unsupported tensor, good python support)
//    * matx (requires cuda to build)
//    * arrayfire (old?)
//    * xtensor (good python support)
//    */
  std::cout << std::endl;
}
