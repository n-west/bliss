#pragma once

#include <bland/ndarray.hpp>
#include <core/cadence.hpp>
#include <core/coarse_channel.hpp>
#include <core/scan.hpp>

namespace bliss {

    coarse_channel excise_dc(coarse_channel cc);

    scan excise_dc(scan sc);

    observation_target excise_dc(observation_target ot);

    cadence excise_dc(cadence ca);

}