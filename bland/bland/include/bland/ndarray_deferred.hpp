#pragma once

#include "ndarray.hpp"

#include <functional>
#include <string_view>
#include <string>
#include <optional>
#include <variant>

namespace bland {

/**
 * ndarray_deferred is backed by a function that returns a tensor. This allows passing around
 * a deferred execution to functions which accept arrays and allowing deferred execution of functins
 * which return arrays.
 * 
 * The function will only execute once with the result cached as an internally held ndarray.
 * 
 * For example, creating ndarray_deferred with a lambda function which reads data from a file to
 * an array will allow passing the ndarray_deferred to functions which accept an array without
 * actually reading the data from disk until that function executes.
*/
class ndarray_deferred {
    public:
    enum class eval_policy {
        memoize, // hold on to a callable and keep the value around once called
        lazy, // always use the callable and never hold on to (memoize) the result
        eager // evaluate callable immediately and store its value
    };

    ndarray_deferred() = default;
    /**
     * create a deferred tensor from a function that will be called when this
     * ndarray_deferred is converted to an ndarray
    */
    ndarray_deferred(std::function<ndarray()> callable, eval_policy policy=eval_policy::memoize);
    
    /**
     * create a deferred tensor that is actual data. This allows treating an ndarray_deferred
     * the same as an ndarray.
    */
    ndarray_deferred(ndarray poa);

    operator ndarray();

    ndarray_deferred to(bland::ndarray::dev device);
    ndarray_deferred to(std::string_view dev_str);

    std::string repr() const;

    private:
    std::shared_ptr<std::variant<ndarray, std::function<ndarray()>>> _deferred_data = nullptr;
    std::optional<ndarray::dev> _device;
    eval_policy _policy = eval_policy::memoize;

};

} // namespace bland
