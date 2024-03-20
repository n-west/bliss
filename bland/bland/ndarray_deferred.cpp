#include "bland/ndarray_deferred.hpp"


#include <fmt/format.h>

using namespace bland;

    // // Constructor from a callable
    // LazyMemoizedArray(std::function<ndarray()> callable)
    //     : _deferred_data(std::make_shared<std::variant<ndarray, std::function<ndarray()>>>(callable)) {}

    // // Constructor from an ndarray
    // LazyMemoizedArray(ndarray poa)
    //     : _deferred_data(std::make_shared<std::variant<ndarray, std::function<ndarray()>>>(poa)) {}


bland::ndarray_deferred::ndarray_deferred(std::function<ndarray()> callable)
    : _deferred_data(std::make_shared<std::variant<ndarray, std::function<ndarray()>>>(callable)) {
}

bland::ndarray_deferred::ndarray_deferred(ndarray pod)
    : _deferred_data(std::make_shared<std::variant<ndarray, std::function<ndarray()>>>(pod)) {
}

ndarray_deferred bland::ndarray_deferred::to(bland::ndarray::dev device) {
    _device = device;
    return *this;
}

ndarray_deferred bland::ndarray_deferred::to(std::string_view dev_str) {
    // It might be convention to make a copy of *this and "send" it to device
    _device = dev_str;
    return *this;
}

    // operator ndarray() {
    //     if (std::holds_alternative<std::function<ndarray()>>(*_deferred_data)) {
    //         *_deferred_data = std::get<std::function<ndarray()>>(*_deferred_data)();
    //     }
    //     return std::get<ndarray>(*_deferred_data);
    // }


bland::ndarray_deferred::operator ndarray() {
    // TODO: to make thread-safe, we need a lock unless it's OK to execute
    // twice in a race
    if (std::holds_alternative<std::function<ndarray()>>(*_deferred_data)) {
        *_deferred_data = std::get<std::function<ndarray()>>(*_deferred_data)();
    }

    auto actual_data = std::get<ndarray>(*_deferred_data);
    if (_device.has_value()) {
        actual_data = actual_data.to(_device.value());
        *_deferred_data = actual_data;
    }
    return actual_data;
}

std::string bland::ndarray_deferred::repr() const {
    if (std::holds_alternative<ndarray>(*_deferred_data)) {
        return std::get<ndarray>(*_deferred_data).repr();
    } else {
        return "ndarray_deferred";
    }
}
