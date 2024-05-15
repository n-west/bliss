
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <pybland.hpp>

#include "core/pyblisscore.hpp"
#include "file_types/pyfile_types.hpp"
#include "flaggers/pyflaggers.hpp"
#include "estimators/pyestimators.hpp"
#include "drift_search/pydrift_search.hpp"

#if BLISS_CUDA
#include <cuda_runtime.h>
#endif

#include <fmt/format.h>

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(pybliss, m) {

    int cuda_runtime_version = 0;
#if BLISS_CUDA
    cudaRuntimeGetVersion(&cuda_runtime_version);
#endif
    std::string cuda_version_string = "none";
    if (cuda_runtime_version != 0) {
        int cuda_major = cuda_runtime_version/1000;
        int cuda_minor = (cuda_runtime_version - 1000 * cuda_major)/10;
        cuda_version_string = fmt::format("{}.{}", cuda_major, cuda_minor);
    }
    m.attr("_cuda_version") = cuda_version_string;

    bind_pycore(m);

    auto bland_module = m.def_submodule("bland", "Breakthrough Listen Arrays with N Dimensions. A multi-device ndarray library based on dlpack.");
    bind_pybland(bland_module);

    auto drift_search_module = m.def_submodule("drift_search", "integrate & search for doppler drifting signals");
    bind_pydrift_search(drift_search_module);

    auto fileio_module = m.def_submodule("io", "File I/O types");
    bind_pyfile_types(fileio_module);

    auto estimators_module = m.def_submodule("estimators", "Estimators");
    bind_pyestimators(estimators_module);

    auto flaggers_module = m.def_submodule("flaggers", "Flaggers");
    bind_pyflaggers(flaggers_module);

    

}
