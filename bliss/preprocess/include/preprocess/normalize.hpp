#pragma once

#include <bland/ndarray.hpp>
#include <core/cadence.hpp>
#include <core/coarse_channel.hpp>
#include <core/scan.hpp>

namespace bliss {

    coarse_channel normalize(coarse_channel cc);

    scan normalize(scan sc);

    observation_target normalize(observation_target ot);

    cadence normalize(cadence ca);

}