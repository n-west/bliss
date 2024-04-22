#include "bland/ndarray_deferred.hpp"


#include <fmt/format.h>

using namespace bland;


bland::ndarray_deferred::ndarray_deferred(std::function<ndarray()> callable, ndarray_deferred::eval_policy policy)
    : _deferred_data(std::make_shared<std::variant<ndarray, std::function<ndarray()>>>(callable)), _policy(policy) {
    if (_policy == eval_policy::eager) {
        if (std::holds_alternative<std::function<ndarray()>>(*_deferred_data)) {
            *_deferred_data = std::get<std::function<ndarray()>>(*_deferred_data)();
        }
    }
}

bland::ndarray_deferred::ndarray_deferred(ndarray pod)
    : _deferred_data(std::make_shared<std::variant<ndarray, std::function<ndarray()>>>(pod)) {
}

ndarray_deferred bland::ndarray_deferred::to(bland::ndarray::dev device) {
    _device = device;
    if (std::holds_alternative<ndarray>(*_deferred_data)) {
        *_deferred_data = std::get<ndarray>(*_deferred_data).to(_device.value());
    }
    return *this;
}

ndarray_deferred bland::ndarray_deferred::to(std::string_view dev_str) {
    // It might be convention to make a copy of *this and "send" it to device
    _device = dev_str;
    return *this;
}

bland::ndarray_deferred::operator ndarray() {
    // TODO: to make thread-safe, we need a lock unless it's OK to execute
    // twice in a race
    bland::ndarray actual_data;
    if (std::holds_alternative<std::function<ndarray()>>(*_deferred_data)) {
        actual_data = std::get<std::function<ndarray()>>(*_deferred_data)();
    } else {
        actual_data = std::get<ndarray>(*_deferred_data);
    }

    if (_policy == eval_policy::memoize) {
        *_deferred_data = actual_data;
    }

    if (_device.has_value()) {
        actual_data = actual_data.to(_device.value());
    }
    return actual_data;
}

std::string bland::ndarray_deferred::repr() const {
    if (std::holds_alternative<ndarray>(*_deferred_data)) {
        return std::get<ndarray>(*_deferred_data).repr();
    } else {
        std::string eval_mode_str;
        if (_policy == eval_policy::eager) {
            eval_mode_str = "eager";
        } else if (_policy == eval_policy::lazy) {
            eval_mode_str = "lazy";
        } else {
            eval_mode_str = "memoize";
        }
        return fmt::format("ndarray_deferred (mode={})", eval_mode_str);
    }
}
