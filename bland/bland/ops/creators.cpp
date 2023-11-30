
#include <bland/creators.hpp>

#include <bland/ndarray.hpp>

#include <random>
#include <stdexcept>
#include <type_traits>

using namespace bland;

template <typename START_TYPE, typename END_TYPE, typename STEP_TYPE>
ndarray bland::arange(START_TYPE start, END_TYPE end, STEP_TYPE step, DLDataType dtype, DLDevice device) {
    auto number_steps  = static_cast<int64_t>((end - start) / step);
    auto aranged_array = ndarray({number_steps}, dtype, device);

    switch (dtype.code) {
    case kDLFloat: {
        switch (dtype.bits) {
        case 32: {
            auto data = aranged_array.data_ptr<float>();
            for (int ii = 0; ii < number_steps; ++ii) {
                data[ii] = start + (step * ii);
            }
        } break;
        case 64: {
            auto data = aranged_array.data_ptr<double>();
            for (int ii = 0; ii < number_steps; ++ii) {
                data[ii] = start + (step * ii);
            }
        } break;
        default:
            throw std::runtime_error("Unsupported float bitwidth");
        }
    } break;
    case kDLInt: {
        switch (dtype.bits) {
        case 8: {
            auto data = aranged_array.data_ptr<int8_t>();
            for (int ii = 0; ii < number_steps; ++ii) {
                data[ii] = start + (step * ii);
            }
        } break;
        case 16: {
            auto data = aranged_array.data_ptr<int16_t>();
            for (int ii = 0; ii < number_steps; ++ii) {
                data[ii] = start + (step * ii);
            }
        } break;
        case 32: {
            auto data = aranged_array.data_ptr<int32_t>();
            for (int ii = 0; ii < number_steps; ++ii) {
                data[ii] = start + (step * ii);
            }
        } break;
        case 64: {
            auto data = aranged_array.data_ptr<int64_t>();
            for (int ii = 0; ii < number_steps; ++ii) {
                data[ii] = start + (step * ii);
            }
        } break;
        default:
            throw std::runtime_error("Unsupported int bitwidth");
        }
    } break;
    case kDLUInt: {
        switch (dtype.bits) {
        case 8: {
            auto data = aranged_array.data_ptr<uint8_t>();
            for (int ii = 0; ii < number_steps; ++ii) {
                data[ii] = start + (step * ii);
            }
        } break;
        case 16: {
            auto data = aranged_array.data_ptr<uint16_t>();
            for (int ii = 0; ii < number_steps; ++ii) {
                data[ii] = start + (step * ii);
            }
        } break;
        case 32: {
            auto data = aranged_array.data_ptr<uint32_t>();
            for (int ii = 0; ii < number_steps; ++ii) {
                data[ii] = start + (step * ii);
            }
        } break;
        case 64: {
            auto data = aranged_array.data_ptr<uint64_t>();
            for (int ii = 0; ii < number_steps; ++ii) {
                data[ii] = start + (step * ii);
            }
        } break;
        default:
            throw std::runtime_error("Unsupported uint bitwidth");
        }
    } break;
    default:
        // auto err = fmt::format("Unsupported datatype code {}", dtype.code);
        // throw std::runtime_error(err);
        throw std::runtime_error("Unsupported datatype code");
    }
    return aranged_array;
}

template ndarray bland::arange<int8_t, int8_t, int8_t>(int8_t     start,
                                                       int8_t     end,
                                                       int8_t     step,
                                                       DLDataType dtype,
                                                       DLDevice   device = default_device);
template ndarray bland::arange<int16_t, int16_t, int16_t>(int16_t    start,
                                                          int16_t    end,
                                                          int16_t    step,
                                                          DLDataType dtype,
                                                          DLDevice   device = default_device);
template ndarray bland::arange<int32_t, int32_t, int32_t>(int32_t    start,
                                                          int32_t    end,
                                                          int32_t    step,
                                                          DLDataType dtype,
                                                          DLDevice   device = default_device);
template ndarray bland::arange<int64_t, int64_t, int64_t>(int64_t    start,
                                                          int64_t    end,
                                                          int64_t    step,
                                                          DLDataType dtype,
                                                          DLDevice   device = default_device);
template ndarray bland::arange<uint8_t, uint8_t, uint8_t>(uint8_t    start,
                                                          uint8_t    end,
                                                          uint8_t    step,
                                                          DLDataType dtype,
                                                          DLDevice   device = default_device);
template ndarray bland::arange<uint16_t, uint16_t, uint16_t>(uint16_t   start,
                                                             uint16_t   end,
                                                             uint16_t   step,
                                                             DLDataType dtype,
                                                             DLDevice   device = default_device);
template ndarray bland::arange<uint32_t, uint32_t, uint32_t>(uint32_t   start,
                                                             uint32_t   end,
                                                             uint32_t   step,
                                                             DLDataType dtype,
                                                             DLDevice   device = default_device);
template ndarray bland::arange<uint64_t, uint64_t, uint64_t>(uint64_t   start,
                                                             uint64_t   end,
                                                             uint64_t   step,
                                                             DLDataType dtype,
                                                             DLDevice   device = default_device);
template ndarray bland::arange<float, float, float>(float      start,
                                                    float      end,
                                                    float      step,
                                                    DLDataType dtype,
                                                    DLDevice   device = default_device);
template ndarray bland::arange<double, double, double>(double     start,
                                                       double     end,
                                                       double     step,
                                                       DLDataType dtype,
                                                       DLDevice   device = default_device);
// Having the triple-template makes things pretty flexible but is a bit of a PITA to instantiate...
template ndarray bland::arange<int32_t, int32_t, uint32_t>(int32_t    start,
                                                           int32_t    end,
                                                           uint32_t   step,
                                                           DLDataType dtype,
                                                           DLDevice   device = default_device);

template <typename START_TYPE, typename END_TYPE, typename STEP_TYPE>
ndarray bland::arange(START_TYPE start, END_TYPE end, STEP_TYPE step, DLDevice device) {
    DLDataType dtype;
    if constexpr (std::is_same<START_TYPE, float>::value) {
        dtype = DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32, .lanes = 1};
    } else if constexpr (std::is_same<START_TYPE, double>::value) {
        dtype = DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 64, .lanes = 1};
    } else if constexpr (std::is_same<START_TYPE, int8_t>::value) {
        dtype = DLDataType{.code = DLDataTypeCode::kDLInt, .bits = 8, .lanes = 1};
    } else if constexpr (std::is_same<START_TYPE, int16_t>::value) {
        dtype = DLDataType{.code = DLDataTypeCode::kDLInt, .bits = 16, .lanes = 1};
    } else if constexpr (std::is_same<START_TYPE, int32_t>::value) {
        dtype = DLDataType{.code = DLDataTypeCode::kDLInt, .bits = 32, .lanes = 1};
    } else if constexpr (std::is_same<START_TYPE, int64_t>::value) {
        dtype = DLDataType{.code = DLDataTypeCode::kDLInt, .bits = 64, .lanes = 1};
    } else if constexpr (std::is_same<START_TYPE, uint8_t>::value) {
        dtype = DLDataType{.code = DLDataTypeCode::kDLUInt, .bits = 8, .lanes = 1};
    } else if constexpr (std::is_same<START_TYPE, uint16_t>::value) {
        dtype = DLDataType{.code = DLDataTypeCode::kDLUInt, .bits = 16, .lanes = 1};
    } else if constexpr (std::is_same<START_TYPE, uint32_t>::value) {
        dtype = DLDataType{.code = DLDataTypeCode::kDLUInt, .bits = 32, .lanes = 1};
    } else if constexpr (std::is_same<START_TYPE, uint64_t>::value) {
        dtype = DLDataType{.code = DLDataTypeCode::kDLUInt, .bits = 64, .lanes = 1};
    } else {
        throw std::runtime_error("arange called without an explicit datatype to generate an unsupported ndarray dtype");
    }

    return arange(start, end, step, dtype, device);
}

template ndarray
bland::arange<int8_t, int8_t, int8_t>(int8_t start, int8_t end, int8_t step, DLDevice device = default_device);
template ndarray
bland::arange<int16_t, int16_t, int16_t>(int16_t start, int16_t end, int16_t step, DLDevice device = default_device);
template ndarray
bland::arange<int32_t, int32_t, int32_t>(int32_t start, int32_t end, int32_t step, DLDevice device = default_device);
template ndarray
bland::arange<int64_t, int64_t, int64_t>(int64_t start, int64_t end, int64_t step, DLDevice device = default_device);
template ndarray
bland::arange<uint8_t, uint8_t, uint8_t>(uint8_t start, uint8_t end, uint8_t step, DLDevice device = default_device);
template ndarray bland::arange<uint16_t, uint16_t, uint16_t>(uint16_t start,
                                                             uint16_t end,
                                                             uint16_t step,
                                                             DLDevice device = default_device);
template ndarray bland::arange<uint32_t, uint32_t, uint32_t>(uint32_t start,
                                                             uint32_t end,
                                                             uint32_t step,
                                                             DLDevice device = default_device);
template ndarray bland::arange<uint64_t, uint64_t, uint64_t>(uint64_t start,
                                                             uint64_t end,
                                                             uint64_t step,
                                                             DLDevice device = default_device);
template ndarray
bland::arange<float, float, float>(float start, float end, float step, DLDevice device = default_device);
template ndarray
bland::arange<double, double, double>(double start, double end, double step, DLDevice device = default_device);

template <typename T>
ndarray bland::linspace(T start, T end, size_t number_steps, DLDataType dtype, DLDevice device) {
    auto aranged_array = ndarray({static_cast<int64_t>(number_steps)}, dtype, device);
    auto step          = (end - start) / number_steps;

    switch (dtype.code) {
    case kDLFloat: {
        switch (dtype.bits) {
        case 32: {
            auto data = aranged_array.data_ptr<float>();
            for (int ii = 0; ii < number_steps; ++ii) {
                data[ii] = start + (step * ii);
            }
        } break;
        case 64: {
            auto data = aranged_array.data_ptr<double>();
            for (int ii = 0; ii < number_steps; ++ii) {
                data[ii] = start + (step * ii);
            }
        } break;
        default:
            throw std::runtime_error("Unsupported float bitwidth");
        }
    } break;
    case kDLInt: {
        switch (dtype.bits) {
        case 8: {
            auto data = aranged_array.data_ptr<int8_t>();
            for (int ii = 0; ii < number_steps; ++ii) {
                data[ii] = start + (step * ii);
            }
        } break;
        case 16: {
            auto data = aranged_array.data_ptr<int16_t>();
            for (int ii = 0; ii < number_steps; ++ii) {
                data[ii] = start + (step * ii);
            }
        } break;
        case 32: {
            auto data = aranged_array.data_ptr<int32_t>();
            for (int ii = 0; ii < number_steps; ++ii) {
                data[ii] = start + (step * ii);
            }
        } break;
        case 64: {
            auto data = aranged_array.data_ptr<int64_t>();
            for (int ii = 0; ii < number_steps; ++ii) {
                data[ii] = start + (step * ii);
            }
        } break;
        default:
            throw std::runtime_error("Unsupported int bitwidth");
        }
    } break;
    case kDLUInt: {
        switch (dtype.bits) {
        case 8: {
            auto data = aranged_array.data_ptr<uint8_t>();
            for (int ii = 0; ii < number_steps; ++ii) {
                data[ii] = start + (step * ii);
            }
        } break;
        case 16: {
            auto data = aranged_array.data_ptr<uint16_t>();
            for (int ii = 0; ii < number_steps; ++ii) {
                data[ii] = start + (step * ii);
            }
        } break;
        case 32: {
            auto data = aranged_array.data_ptr<uint32_t>();
            for (int ii = 0; ii < number_steps; ++ii) {
                data[ii] = start + (step * ii);
            }
        } break;
        case 64: {
            auto data = aranged_array.data_ptr<uint64_t>();
            for (int ii = 0; ii < number_steps; ++ii) {
                data[ii] = start + (step * ii);
            }
        } break;
        default:
            throw std::runtime_error("Unsupported uint bitwidth");
        }
    } break;
    default:
        // auto err = fmt::format("Unsupported datatype code {}", dtype.code);
        // throw std::runtime_error(err);
        // std::cout << "dtype code is " << std::string((char)(dtype.code)) << std::endl;
        throw std::runtime_error("Unsupported datatype code");
    }
    return aranged_array;
}

template ndarray bland::linspace<int8_t>(int8_t     start,
                                         int8_t     end,
                                         size_t     number_steps,
                                         DLDataType dtype,
                                         DLDevice   device = default_device);
template ndarray bland::linspace<int16_t>(int16_t    start,
                                          int16_t    end,
                                          size_t     number_steps,
                                          DLDataType dtype,
                                          DLDevice   device = default_device);
template ndarray bland::linspace<int32_t>(int32_t    start,
                                          int32_t    end,
                                          size_t     number_steps,
                                          DLDataType dtype,
                                          DLDevice   device = default_device);
template ndarray bland::linspace<int64_t>(int64_t    start,
                                          int64_t    end,
                                          size_t     number_steps,
                                          DLDataType dtype,
                                          DLDevice   device = default_device);
template ndarray bland::linspace<uint8_t>(uint8_t    start,
                                          uint8_t    end,
                                          size_t     number_steps,
                                          DLDataType dtype,
                                          DLDevice   device = default_device);
template ndarray bland::linspace<uint16_t>(uint16_t   start,
                                           uint16_t   end,
                                           size_t     number_steps,
                                           DLDataType dtype,
                                           DLDevice   device = default_device);
template ndarray bland::linspace<uint32_t>(uint32_t   start,
                                           uint32_t   end,
                                           size_t     number_steps,
                                           DLDataType dtype,
                                           DLDevice   device = default_device);
template ndarray bland::linspace<uint64_t>(uint64_t   start,
                                           uint64_t   end,
                                           size_t     number_steps,
                                           DLDataType dtype,
                                           DLDevice   device = default_device);
template ndarray
bland::linspace<float>(float start, float end, size_t number_steps, DLDataType dtype, DLDevice device = default_device);
template ndarray bland::linspace<double>(double     start,
                                         double     end,
                                         size_t     number_steps,
                                         DLDataType dtype,
                                         DLDevice   device = default_device);

template <typename T>
ndarray bland::linspace(T start, T end, size_t number_steps, DLDevice device) {
    DLDataType dtype;
    if constexpr (std::is_same<T, float>::value) {
        dtype = DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32, .lanes = 1};
    } else if constexpr (std::is_same<T, double>::value) {
        dtype = DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 64, .lanes = 1};
    } else if constexpr (std::is_same<T, int8_t>::value) {
        dtype = DLDataType{.code = DLDataTypeCode::kDLInt, .bits = 8, .lanes = 1};
    } else if constexpr (std::is_same<T, int16_t>::value) {
        dtype = DLDataType{.code = DLDataTypeCode::kDLInt, .bits = 16, .lanes = 1};
    } else if constexpr (std::is_same<T, int32_t>::value) {
        dtype = DLDataType{.code = DLDataTypeCode::kDLInt, .bits = 32, .lanes = 1};
    } else if constexpr (std::is_same<T, int64_t>::value) {
        dtype = DLDataType{.code = DLDataTypeCode::kDLInt, .bits = 64, .lanes = 1};
    } else if constexpr (std::is_same<T, uint8_t>::value) {
        dtype = DLDataType{.code = DLDataTypeCode::kDLUInt, .bits = 8, .lanes = 1};
    } else if constexpr (std::is_same<T, uint16_t>::value) {
        dtype = DLDataType{.code = DLDataTypeCode::kDLUInt, .bits = 16, .lanes = 1};
    } else if constexpr (std::is_same<T, uint32_t>::value) {
        dtype = DLDataType{.code = DLDataTypeCode::kDLUInt, .bits = 32, .lanes = 1};
    } else if constexpr (std::is_same<T, uint64_t>::value) {
        dtype = DLDataType{.code = DLDataTypeCode::kDLUInt, .bits = 64, .lanes = 1};
    } else {
        throw std::runtime_error("arange called without an explicit datatype to generate an unsupported ndarray dtype");
    }

    return linspace(start, end, number_steps, dtype, device);
}

template ndarray
bland::linspace<int8_t>(int8_t start, int8_t end, size_t number_steps, DLDevice device = default_device);
template ndarray
bland::linspace<int16_t>(int16_t start, int16_t end, size_t number_steps, DLDevice device = default_device);
template ndarray
bland::linspace<int32_t>(int32_t start, int32_t end, size_t number_steps, DLDevice device = default_device);
template ndarray
bland::linspace<int64_t>(int64_t start, int64_t end, size_t number_steps, DLDevice device = default_device);
template ndarray
bland::linspace<uint8_t>(uint8_t start, uint8_t end, size_t number_steps, DLDevice device = default_device);
template ndarray
bland::linspace<uint16_t>(uint16_t start, uint16_t end, size_t number_steps, DLDevice device = default_device);
template ndarray
bland::linspace<uint32_t>(uint32_t start, uint32_t end, size_t number_steps, DLDevice device = default_device);
template ndarray
bland::linspace<uint64_t>(uint64_t start, uint64_t end, size_t number_steps, DLDevice device = default_device);
template ndarray bland::linspace<float>(float start, float end, size_t number_steps, DLDevice device = default_device);
template ndarray
bland::linspace<double>(double start, double end, size_t number_steps, DLDevice device = default_device);

ndarray bland::zeros(std::vector<int64_t> shape, DLDataType dtype, DLDevice device) {
    return ndarray(shape, 0, dtype, device);
}

ndarray bland::ones(std::vector<int64_t> shape, DLDataType dtype, DLDevice device) {
    return ndarray(shape, 1, dtype, device);
}

template <typename T, typename D>
void initialize_randn(T *data, int64_t numel, D distribution) {
    std::random_device rd{};
    std::mt19937       gen{rd()};
    // We know this is a dense array
    for (int n = 0; n < numel; ++n) {
        data[n] = distribution(gen);
    }
}

template <typename T>
ndarray bland::rand_normal(std::vector<int64_t> shape, T mean, T stddev, DLDataType dtype, DLDevice device) {
    auto new_array = ndarray(shape, dtype, device);
    if (dtype.code != DLDataTypeCode::kDLFloat) {
        throw std::runtime_error("Cannot generate normal distribution for non-floating point type");
    }

    if (dtype.code == DLDataTypeCode::kDLFloat && dtype.bits == 32) {
        std::normal_distribution<float> d{static_cast<float>(mean), static_cast<float>(stddev)};
        initialize_randn(new_array.data_ptr<float>(), new_array.numel(), d);
    } else if (dtype.code == DLDataTypeCode::kDLFloat && dtype.bits == 64) {
        std::normal_distribution<double> d{static_cast<double>(mean), static_cast<double>(stddev)};
        initialize_randn(new_array.data_ptr<double>(), new_array.numel(), d);
    } else {
        throw std::runtime_error("rand_normal: an unexpected error occured, probably unknown bit width");
    }

    return new_array;
}

template ndarray
bland::rand_normal(std::vector<int64_t> shape, float mean, float variance, DLDataType dtype, DLDevice device);
template ndarray
bland::rand_normal(std::vector<int64_t> shape, double mean, double variance, DLDataType dtype, DLDevice device);

template <typename T>
ndarray bland::rand_uniform(std::vector<int64_t> shape, T low, T high, DLDataType dtype, DLDevice device) {
    auto new_array = ndarray(shape, dtype, device);

    if (dtype.code == DLDataTypeCode::kDLFloat) {
        if (dtype.bits == 32) {
            std::uniform_real_distribution<float> d{static_cast<float>(low), static_cast<float>(high)};
            initialize_randn(new_array.data_ptr<float>(), new_array.numel(), d);
        } else if (dtype.bits == 64) {
            std::uniform_real_distribution<double> d{static_cast<double>(low), static_cast<double>(high)};
            initialize_randn(new_array.data_ptr<double>(), new_array.numel(), d);
        }
    } else if (dtype.code == DLDataTypeCode::kDLInt) {
        if (dtype.bits == 8) {
            std::uniform_int_distribution<int8_t> d{static_cast<int8_t>(low), static_cast<int8_t>(high)};
            initialize_randn(new_array.data_ptr<int8_t>(), new_array.numel(), d);
        } else if (dtype.bits == 16) {
            std::uniform_int_distribution<int16_t> d{static_cast<int16_t>(low), static_cast<int16_t>(high)};
            initialize_randn(new_array.data_ptr<int16_t>(), new_array.numel(), d);
        } else if (dtype.bits == 32) {
            std::uniform_int_distribution<int32_t> d{static_cast<int32_t>(low), static_cast<int32_t>(high)};
            initialize_randn(new_array.data_ptr<int32_t>(), new_array.numel(), d);
        } else if (dtype.bits == 64) {
            std::uniform_int_distribution<int64_t> d{static_cast<int64_t>(low), static_cast<int64_t>(high)};
            initialize_randn(new_array.data_ptr<int64_t>(), new_array.numel(), d);
        }
    } else if (dtype.code == DLDataTypeCode::kDLUInt) {
        if (dtype.bits == 8) {
            std::uniform_int_distribution<uint8_t> d{static_cast<uint8_t>(low), static_cast<uint8_t>(high)};
            initialize_randn(new_array.data_ptr<uint8_t>(), new_array.numel(), d);
        } else if (dtype.bits == 16) {
            std::uniform_int_distribution<uint16_t> d{static_cast<uint16_t>(low), static_cast<uint16_t>(high)};
            initialize_randn(new_array.data_ptr<uint16_t>(), new_array.numel(), d);
        } else if (dtype.bits == 32) {
            std::uniform_int_distribution<uint32_t> d{static_cast<uint32_t>(low), static_cast<uint32_t>(high)};
            initialize_randn(new_array.data_ptr<uint32_t>(), new_array.numel(), d);
        } else if (dtype.bits == 64) {
            std::uniform_int_distribution<uint64_t> d{static_cast<uint64_t>(low), static_cast<uint64_t>(high)};
            initialize_randn(new_array.data_ptr<uint64_t>(), new_array.numel(), d);
        }
    }

    return new_array;
}

template ndarray
bland::rand_uniform(std::vector<int64_t> shape, float low, float high, DLDataType dtype, DLDevice device);
template ndarray
bland::rand_uniform(std::vector<int64_t> shape, double low, double high, DLDataType dtype, DLDevice device);
template ndarray
bland::rand_uniform(std::vector<int64_t> shape, int8_t low, int8_t high, DLDataType dtype, DLDevice device);
template ndarray
bland::rand_uniform(std::vector<int64_t> shape, int16_t low, int16_t high, DLDataType dtype, DLDevice device);
template ndarray
bland::rand_uniform(std::vector<int64_t> shape, int32_t low, int32_t high, DLDataType dtype, DLDevice device);
template ndarray
bland::rand_uniform(std::vector<int64_t> shape, int64_t low, int64_t high, DLDataType dtype, DLDevice device);
template ndarray
bland::rand_uniform(std::vector<int64_t> shape, uint8_t low, uint8_t high, DLDataType dtype, DLDevice device);
template ndarray
bland::rand_uniform(std::vector<int64_t> shape, uint16_t low, uint16_t high, DLDataType dtype, DLDevice device);
template ndarray
bland::rand_uniform(std::vector<int64_t> shape, uint32_t low, uint32_t high, DLDataType dtype, DLDevice device);
template ndarray
bland::rand_uniform(std::vector<int64_t> shape, uint64_t low, uint64_t high, DLDataType dtype, DLDevice device);
