#pragma once

#include "bland/ndarray.hpp"

#include <fmt/core.h>

#include <stdexcept>

namespace bland {


enum class Constraints : uint64_t {
    None = 0,
    IdentityType = 1 << 0, // 1
    NoFloat = 1 << 1, // 2
    NoInt = 1 << 2, // 4
    NoUInt = 1 << 3, // 8
};
// Enable bitwise operators for the enum
constexpr Constraints operator|(Constraints lhs, Constraints rhs) {
    return static_cast<Constraints>(
        static_cast<uint64_t>(lhs) | static_cast<uint64_t>(rhs)
    );
}

constexpr Constraints& operator|=(Constraints& lhs, Constraints rhs) {
    lhs = lhs | rhs;
    return lhs;
}

constexpr bool operator&(Constraints lhs, Constraints rhs) {
    return static_cast<bool>(
        static_cast<int64_t>(lhs) & static_cast<int64_t>(rhs)
    );
}

template <Constraints constraints>
constexpr bool has_constraint(Constraints constraint) {
    return (constraints & constraint);
}

/**
 * Dispatch operations to implementations templated on actual datatypes
 * by reading the runtime ndarray dtype and translating that to compile-time functions called with
 * template arguments based on the dtype.
 * 
 * This is highly verbose, but conceptually simple even though the syntax is somewhat advanced
 * to get datatypes, template substitution, and perfect argument forwarding.
 * 
 * The main benefit of this is keeping the overhead of runtime dispatching pretty minimal (two switch statements
 * per input) and the remaining work is all compile-time optimized work
 *
 */


/**
 * out = f(a, b) where a, b, and c are ndarrays.
 * 
 * This requires three type deductions and passes through an underlying impl function with 3 template args
*/
template <typename F, typename Out, typename A, typename ...Args>
ndarray dispatch(ndarray &out, const ndarray &a, const ndarray &b, Args... args) {
    auto dtype = b.dtype();

    switch (dtype.code) {
    case kDLFloat: {
        switch (dtype.bits) {
        case 32:
            return F::template call<Out, A, float>(out, a, b, std::forward<Args>(args)...);
        // case 64:
        //     return F::template call<Out, A, double>(out, a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported float bitwidth");
        }
    }
    case kDLInt: {
        switch (dtype.bits) {
        // case 8:
        //     return F::template call<Out, A, int8_t>(out, a, b, std::forward<Args>(args)...);
        // case 16:
        //     return F::template call<Out, A, int16_t>(out, a, b, std::forward<Args>(args)...);
        case 32:
            return F::template call<Out, A, int32_t>(out, a, b, std::forward<Args>(args)...);
        // case 64:
        //     return F::template call<Out, A, int64_t>(out, a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported int bitwidth");
        }
    }
    case kDLUInt: {
        switch (dtype.bits) {
        case 8:
            return F::template call<Out, A, uint8_t>(out, a, b, std::forward<Args>(args)...);
        // case 16:
        //     return F::template call<Out, A, uint16_t>(out, a, b, std::forward<Args>(args)...);
        case 32:
            return F::template call<Out, A, uint32_t>(out, a, b, std::forward<Args>(args)...);
        // case 64:
        //     return F::template call<Out, A, uint64_t>(out, a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported uint bitwidth");
        }
    }
    default:
        throw std::runtime_error("Unsupported datatype code");
    }
}

template <typename F, typename Out, typename ...Args>
ndarray dispatch(ndarray &out, const ndarray &a, const ndarray &b, Args... args) {
    auto dtype = a.dtype();

    switch (dtype.code) {
    case kDLFloat: {
        switch (dtype.bits) {
        case 32:
            return dispatch<F, Out, float>(out, a, b, std::forward<Args>(args)...);
        // case 64:
        //     return dispatch<F, Out, double>(out, a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported float bitwidth");
        }
    }
    case kDLInt: {
        switch (dtype.bits) {
        // case 8:
        //     return dispatch<F, Out, int8_t>(out, a, b, std::forward<Args>(args)...);
        // case 16:
        //     return dispatch<F, Out, int16_t>(out, a, b, std::forward<Args>(args)...);
        case 32:
            return dispatch<F, Out, int32_t>(out, a, b, std::forward<Args>(args)...);
        // case 64:
        //     return dispatch<F, Out, int64_t>(out, a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported int bitwidth");
        }
    }
    case kDLUInt: {
        switch (dtype.bits) {
        case 8:
            return dispatch<F, Out, uint8_t>(out, a, b, std::forward<Args>(args)...);
        // case 16:
        //     return dispatch<F, Out, uint16_t>(out, a, b, std::forward<Args>(args)...);
        case 32:
            return dispatch<F, Out, uint32_t>(out, a, b, std::forward<Args>(args)...);
        // case 64:
        //     return dispatch<F, Out, uint64_t>(out, a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported uint bitwidth");
        }
    }
    default:
        throw std::runtime_error("Unsupported datatype code");
    }
}

/**
 * out = f(a, b) where a and b are ndarray
*/
template <typename F, typename ...Args>
ndarray dispatch(ndarray &out, const ndarray &a, const ndarray &b, Args... args) {
    auto dtype = out.dtype();

    switch (dtype.code) {
    case kDLFloat: {
        switch (dtype.bits) {
        case 32:
            return dispatch<F, float>(out, a, b, std::forward<Args>(args)...);
        // case 64:
        //     return dispatch<F, double>(out, a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported float bitwidth");
        }
    }
    case kDLInt: {
        switch (dtype.bits) {
        // case 8:
        //     return dispatch<F, int8_t>(out, a, b, std::forward<Args>(args)...);
        // case 16:
        //     return dispatch<F, int16_t>(out, a, b, std::forward<Args>(args)...);
        case 32:
            return dispatch<F, int32_t>(out, a, b, std::forward<Args>(args)...);
        // case 64:
        //     return dispatch<F, int64_t>(out, a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported int bitwidth");
        }
    }
    case kDLUInt: {
        switch (dtype.bits) {
        case 8:
            return dispatch<F, uint8_t>(out, a, b, std::forward<Args>(args)...);
        // case 16:
        //     return dispatch<F, uint16_t>(out, a, b, std::forward<Args>(args)...);
        case 32:
            return dispatch<F, uint32_t>(out, a, b, std::forward<Args>(args)...);
        // case 64:
        //     return dispatch<F, uint64_t>(out, a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported uint bitwidth");
        }
    }
    default:
        throw std::runtime_error("Unsupported datatype code");
    }
}


/**
 * out = f(a, b) where a, b, and c are ndarrays.
 * 
 * This requires three type deductions and passes through an underlying impl function with 3 template args
*/
template <Constraints constraints, typename F, typename Out, typename A, typename ...Args>
ndarray constrained_dispatch(ndarray &out, const ndarray &a, const ndarray &b, Args... args) {
    auto dtype = b.dtype();

    if (dtype.code == kDLFloat) {
        if constexpr(!has_constraint<constraints>(Constraints::NoFloat)) {
            switch (dtype.bits) {
            case 32:
                return F::template call<Out, A, float>(out, a, b, std::forward<Args>(args)...);
            // case 64:
            //     return F::template call<Out, A, double>(out, a, b, std::forward<Args>(args)...);
            default:
                throw std::runtime_error("Unsupported float bitwidth");
            }
        } else {
            throw std::runtime_error("Function does not support float dtypes");
        }
    } else if (dtype.code == kDLInt) {
        switch (dtype.bits) {
        // case 8:
        //     return F::template call<Out, A, int8_t>(out, a, b, std::forward<Args>(args)...);
        // case 16:
        //     return F::template call<Out, A, int16_t>(out, a, b, std::forward<Args>(args)...);
        case 32:
            return F::template call<Out, A, int32_t>(out, a, b, std::forward<Args>(args)...);
        // case 64:
        //     return F::template call<Out, A, int64_t>(out, a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported int bitwidth");
        }
    } else if (dtype.code == kDLUInt) {
        switch (dtype.bits) {
        case 8:
            return F::template call<Out, A, uint8_t>(out, a, b, std::forward<Args>(args)...);
        // case 16:
        //     return F::template call<Out, A, uint16_t>(out, a, b, std::forward<Args>(args)...);
        case 32:
            return F::template call<Out, A, uint32_t>(out, a, b, std::forward<Args>(args)...);
        // case 64:
        //     return F::template call<Out, A, uint64_t>(out, a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported uint bitwidth");
        }
    } else {
        throw std::runtime_error("Unsupported datatype code");
    }
}

template <Constraints constraints, typename F, typename Out, typename ...Args>
ndarray constrained_dispatch(ndarray &out, const ndarray &a, const ndarray &b, Args... args) {
    auto dtype = a.dtype();

    if (dtype.code == kDLFloat) {
        if constexpr(!has_constraint<constraints>(Constraints::NoFloat)) {
            switch (dtype.bits) {
            case 32:
               return constrained_dispatch<constraints, F, Out, float>(out, a, b, std::forward<Args>(args)...);
            // case 64:
            //     return dispatch<F, Out, double>(out, a, b, std::forward<Args>(args)...);
            default:
               throw std::runtime_error("Unsupported float bitwidth");
            }
        } else {
            throw std::runtime_error("Function does not support float dtypes");
        }
    } else
    if (dtype.code == kDLInt) {
        switch (dtype.bits) {
        // case 8:
        //     return dispatch<F, Out, int8_t>(out, a, b, std::forward<Args>(args)...);
        // case 16:
        //     return dispatch<F, Out, int16_t>(out, a, b, std::forward<Args>(args)...);
        case 32:
            return constrained_dispatch<constraints, F, Out, int32_t>(out, a, b, std::forward<Args>(args)...);
        // case 64:
        //     return dispatch<F, Out, int64_t>(out, a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported int bitwidth");
        }
    } else if (dtype.code == kDLUInt) {
        switch (dtype.bits) {
        case 8:
            return constrained_dispatch<constraints, F, Out, uint8_t>(out, a, b, std::forward<Args>(args)...);
        // case 16:
        //     return dispatch<F, Out, uint16_t>(out, a, b, std::forward<Args>(args)...);
        case 32:
            return constrained_dispatch<constraints, F, Out, uint32_t>(out, a, b, std::forward<Args>(args)...);
        // case 64:
        //     return dispatch<F, Out, uint64_t>(out, a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported uint bitwidth");
        }
    } else {
        throw std::runtime_error("Unsupported datatype code");
    }
}

/**
 * out = f(a, b) where a and b are ndarray
*/
template <Constraints constraints, typename F, typename ...Args>
ndarray constrained_dispatch(ndarray &out, const ndarray &a, const ndarray &b, Args... args) {
    auto dtype = out.dtype();

    if (dtype.code == kDLFloat) {
        if constexpr(!has_constraint<constraints>(Constraints::NoFloat)) {
            switch (dtype.bits) {
            case 32:
                return constrained_dispatch<constraints, F, float>(out, a, b, std::forward<Args>(args)...);
            // case 64:
            //     return dispatch<F, double>(out, a, b, std::forward<Args>(args)...);
            default:
                throw std::runtime_error("Unsupported float bitwidth");
            }
        } else {
            throw std::runtime_error("Function does not support float dtypes");
        }
    } else if (dtype.code == kDLInt) {
        switch (dtype.bits) {
        // case 8:
        //     return dispatch<F, int8_t>(out, a, b, std::forward<Args>(args)...);
        // case 16:
        //     return dispatch<F, int16_t>(out, a, b, std::forward<Args>(args)...);
        case 32:
            return constrained_dispatch<constraints, F, int32_t>(out, a, b, std::forward<Args>(args)...);
        // case 64:
        //     return dispatch<F, int64_t>(out, a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported int bitwidth");
        }
    } else if (dtype.code == kDLUInt) {
        switch (dtype.bits) {
        case 8:
            return constrained_dispatch<constraints, F, uint8_t>(out, a, b, std::forward<Args>(args)...);
        // case 16:
        //     return dispatch<F, uint16_t>(out, a, b, std::forward<Args>(args)...);
        case 32:
            return constrained_dispatch<constraints, F, uint32_t>(out, a, b, std::forward<Args>(args)...);
        // case 64:
        //     return dispatch<F, uint64_t>(out, a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported uint bitwidth");
        }
    } else {
        throw std::runtime_error("Unsupported datatype code");
    }
}



/**
 * out = f(a, b) where a is ndarray and b is a scalar that outputs an ndarray
 * 
 * This requires three type deductions and passes through an underlying impl function with 3 template args
 **/
template <Constraints constraints, typename F, typename Out, typename S, class Op, typename ...Args>
ndarray constrained_dispatch_nd_sc(ndarray &out, const ndarray &a, const S &b, Args... args) {
    auto a_dtype = a.dtype();

    if (a_dtype.code == kDLFloat) {
        if constexpr (!has_constraint<constraints>(Constraints::NoFloat)) {
            switch (a_dtype.bits) {
            case 32:
                return F::template call<Out, float, S, Op>(out, a, b, std::forward<Args>(args)...);
            // case 64:
            //     return F::template call<Out, double, S, Op>(out, a, b, std::forward<Args>(args)...);
            default:
                throw std::runtime_error("Unsupported float bitwidth");
            }
        } else
            throw std::runtime_error("Function does not support float dtype");
        }
    if (a_dtype.code == kDLInt) {
        switch (a_dtype.bits) {
        // case 8:
        //     return F::template call<Out, int8_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        // case 16:
        //     return F::template call<Out, int16_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        case 32:
            return F::template call<Out, int32_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        case 64:
            return F::template call<Out, int64_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported int bitwidth");
        }
    } else if (a_dtype.code == kDLUInt) {
        switch (a_dtype.bits) {
        case 8:
            return F::template call<Out, uint8_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        // case 16:
        //     return F::template call<Out, uint16_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        case 32:
            return F::template call<Out, uint32_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        // case 64:
        //     return F::template call<Out, uint64_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported uint bitwidth");
        }
    } else {
        throw std::runtime_error("Unsupported datatype code");
    }
}


template <Constraints constraints, typename F, typename S, class Op, typename ...Args>
ndarray constrained_dispatch_nd_sc(ndarray &out, const ndarray &a, const S &b, Args... args) {
    auto dtype = out.dtype();

    if (dtype.code == kDLFloat) {
        if constexpr (!has_constraint<constraints>(Constraints::NoFloat)) {
            switch (dtype.bits) {
            case 32:
                return constrained_dispatch_nd_sc<constraints, F, float, S, Op>(out, a, b, std::forward<Args>(args)...);
            // case 64:
            //     return dispatch_new3<F, double, S, Op>(out, a, b, std::forward<Args>(args)...);
            default:
                throw std::runtime_error("Unsupported float bitwidth");
            }
        } else {
            throw std::runtime_error("Function does not support float dtypes");
        }
    } else 
    if (dtype.code == kDLInt) {
        switch (dtype.bits) {
        // case 8:
        //     return dispatch_new3<F, int8_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        // case 16:
        //     return dispatch_new3<F, int16_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        case 32:
            return constrained_dispatch_nd_sc<constraints, F, int32_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        case 64:
            return constrained_dispatch_nd_sc<constraints, F, int64_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported int bitwidth");
        }
    } else if (dtype.code == kDLUInt) {
        switch (dtype.bits) {
        case 8:
            return constrained_dispatch_nd_sc<constraints, F, uint8_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        // case 16:
        //     return dispatch_new3<F, uint16_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        case 32:
            return constrained_dispatch_nd_sc<constraints, F, uint32_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        // case 64:
        //     return dispatch_new3<F, uint64_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported uint bitwidth");
        }
    } else {
        throw std::runtime_error("Unsupported datatype code");
    }
}




/**
 * out = f(a, b) where a is ndarray and b is a scalar that outputs an ndarray
 * 
 * This requires three type deductions and passes through an underlying impl function with 3 template args
 **/
template <typename F, typename Out, typename S, class Op, typename ...Args>
ndarray dispatch_nd_sc(ndarray &out, const ndarray &a, const S &b, Args... args) {
    auto a_dtype = a.dtype();

    switch (a_dtype.code) {

    case kDLFloat: {
        switch (a_dtype.bits) {
        case 32:
            return F::template call<Out, float, S, Op>(out, a, b, std::forward<Args>(args)...);
        // case 64:
        //     return F::template call<Out, double, S, Op>(out, a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported float bitwidth");
        }
    }
    case kDLInt: {
        switch (a_dtype.bits) {
        // case 8:
        //     return F::template call<Out, int8_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        // case 16:
        //     return F::template call<Out, int16_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        case 32:
            return F::template call<Out, int32_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        case 64:
            return F::template call<Out, int64_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported int bitwidth");
        }
    }
    case kDLUInt: {
        switch (a_dtype.bits) {
        case 8:
            return F::template call<Out, uint8_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        // case 16:
        //     return F::template call<Out, uint16_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        case 32:
            return F::template call<Out, uint32_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        // case 64:
        //     return F::template call<Out, uint64_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported uint bitwidth");
        }
    }
    default:
        throw std::runtime_error("Unsupported datatype code");
    }
}


template <typename F, typename S, class Op, typename ...Args>
ndarray dispatch_nd_sc(ndarray &out, const ndarray &a, const S &b, Args... args) {
    auto dtype = out.dtype();

    switch (dtype.code) {

    case kDLFloat: {
        switch (dtype.bits) {
        case 32:
            return dispatch_nd_sc<F, float, S, Op>(out, a, b, std::forward<Args>(args)...);
        // case 64:
        //     return dispatch_new3<F, double, S, Op>(out, a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported float bitwidth");
        }
    }
    case kDLInt: {
        switch (dtype.bits) {
        // case 8:
        //     return dispatch_new3<F, int8_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        // case 16:
        //     return dispatch_new3<F, int16_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        case 32:
            return dispatch_nd_sc<F, int32_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        case 64:
            return dispatch_nd_sc<F, int64_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported int bitwidth");
        }
    }
    case kDLUInt: {
        switch (dtype.bits) {
        case 8:
            return dispatch_nd_sc<F, uint8_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        // case 16:
        //     return dispatch_new3<F, uint16_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        case 32:
            return dispatch_nd_sc<F, uint32_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        // case 64:
        //     return dispatch_new3<F, uint64_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported uint bitwidth");
        }
    }
    default:
        throw std::runtime_error("Unsupported datatype code");
    }
}




/**
 * out = f(b) where c is an ndarray and b is a scalar. (think fill(b))
 * 
 * This requires one type deductions and passes through an underlying impl function with 2 template args
 **/
template <typename F, typename S, typename ...Args>
ndarray dispatch_scalar(ndarray &out, const S &b, Args... args) {
    auto dtype = out.dtype();

    switch (dtype.code) {

    case kDLFloat: {
        switch (dtype.bits) {
        case 32:
            return F::template call<float, S>(out, b, std::forward<Args>(args)...);
        // case 64:
        //     return F::template call<double, S>(out, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported float bitwidth");
        }
    }
    case kDLInt: {
        switch (dtype.bits) {
        // case 8:
        //     return F::template call<int8_t, S>(out, b, std::forward<Args>(args)...);
        // case 16:
        //     return F::template call<int16_t, S>(out, b, std::forward<Args>(args)...);
        case 32:
            return F::template call<int32_t, S>(out, b, std::forward<Args>(args)...);
        // case 64:
        //     return F::template call<int64_t, S>(out, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported int bitwidth");
        }
    }
    case kDLUInt: {
        switch (dtype.bits) {
        case 8:
            return F::template call<uint8_t, S>(out, b, std::forward<Args>(args)...);
        // case 16:
        //     return F::template call<uint16_t, S>(out, b, std::forward<Args>(args)...);
        case 32:
            return F::template call<uint32_t, S>(out, b, std::forward<Args>(args)...);
        // case 64:
        //     return F::template call<uint64_t, S>(out, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported uint bitwidth");
        }
    }
    default:
        throw std::runtime_error("Unsupported datatype code");
    }
}


/**
 * Deduce the first input datatype to a reduction op
*/
/**
 * out = f(a) where a is an array and out must be the same type.
 * 
 * This requires one type deductions and passes through an underlying impl function with 1 template args
 * 
 * Used by
 * * max
 **/
template <class Op, typename... Args>
ndarray dispatch_new4(ndarray &out, const ndarray &a, Args... args) {
    auto dtype = a.dtype();
    auto out_dtype = out.dtype();
    if (dtype != out_dtype) {
        throw std::runtime_error("dispatch_new4: out dtype is not the same as the in dtype");
    }

    switch (dtype.code) {

    case kDLFloat: {
        switch (dtype.bits) {
        case 32:
            return Op::template call<float>(out, a, std::forward<Args>(args)...);
        // case 64:
        //     return Op::template call<double>(out, a, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported float bitwidth");
        }
    }
    case kDLInt: {
        switch (dtype.bits) {
        // case 8:
        //     return Op::template call<int8_t>(out, a, std::forward<Args>(args)...);
        // case 16:
        //     return Op::template call<int16_t>(out, a, std::forward<Args>(args)...);
        case 32:
            return Op::template call<int32_t>(out, a, std::forward<Args>(args)...);
        case 64:
            return Op::template call<int64_t>(out, a, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported int bitwidth");
        }
    }
    case kDLUInt: {
        switch (dtype.bits) {
        case 8:
            return Op::template call<uint8_t>(out, a, std::forward<Args>(args)...);
        // case 16:
        //     return Op::template call<uint16_t>(out, a, std::forward<Args>(args)...);
        case 32:
            return Op::template call<uint32_t>(out, a, std::forward<Args>(args)...);
        // case 64:
        //     return Op::template call<uint64_t>(out, a, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported uint bitwidth");
        }
    }
    default:
        throw std::runtime_error("Unsupported datatype code");
    }
}

/**
 * Deduce the first input datatype to a reduction op
*/
/**
 * out = f(a) where a is an ndarray and returns an ndarray
 * 
 * This requires two type deductions and passes through an underlying impl function with 2 template args
 * 
 * Used by
 * * mean
 * * stddev
 * * standardized_moment
 * * sum
 **/
template <typename Out, class Op, typename... Args>
ndarray dispatch_new(ndarray &out, const ndarray &a, Args... args) {
    auto dtype = a.dtype();

    switch (dtype.code) {

    case kDLFloat: {
        switch (dtype.bits) {
        case 32:
            return Op::template call<Out, float>(out, a, std::forward<Args>(args)...);
        // case 64:
        //     return Op::template call<Out, double>(out, a, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported float bitwidth");
        }
    }
    case kDLInt: {
        switch (dtype.bits) {
        // case 8:
        //     return Op::template call<Out, int8_t>(out, a, std::forward<Args>(args)...);
        // case 16:
        //     return Op::template call<Out, int16_t>(out, a, std::forward<Args>(args)...);
        case 32:
            return Op::template call<Out, int32_t>(out, a, std::forward<Args>(args)...);
        case 64:
            return Op::template call<Out, int64_t>(out, a, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported int bitwidth");
        }
    }
    case kDLUInt: {
        switch (dtype.bits) {
        case 8:
            return Op::template call<Out, uint8_t>(out, a, std::forward<Args>(args)...);
        // case 16:
        //     return Op::template call<Out, uint16_t>(out, a, std::forward<Args>(args)...);
        case 32:
            return Op::template call<Out, uint32_t>(out, a, std::forward<Args>(args)...);
        // case 64:
        //     return Op::template call<Out, uint64_t>(out, a, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported uint bitwidth");
        }
    }
    default:
        throw std::runtime_error("Unsupported datatype code");
    }
}

/**
 * out = f(a) where out and a are ndarray
*/
template <class Op, typename... Args>
ndarray dispatch_new(ndarray &out, const ndarray &a, Args... args) {
    auto dtype = out.dtype();

    switch (dtype.code) {

    case kDLFloat: {
        switch (dtype.bits) {
        case 32:
            return dispatch_new<float, Op>(out, a, std::forward<Args>(args)...);
        // case 64:
        //     return dispatch_new<double, Op>(out, a, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported float bitwidth");
        }
    }
    case kDLInt: {
        switch (dtype.bits) {
        // case 8:
        //     return dispatch_new<int8_t, Op>(out, a, std::forward<Args>(args)...);
        // case 16:
        //     return dispatch_new<int16_t, Op>(out, a, std::forward<Args>(args)...);
        case 32:
            return dispatch_new<int32_t, Op>(out, a, std::forward<Args>(args)...);
        case 64:
            return dispatch_new<int64_t, Op>(out, a, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported int bitwidth");
        }
    }
    case kDLUInt: {
        switch (dtype.bits) {
        case 8:
            return dispatch_new<uint8_t, Op>(out, a, std::forward<Args>(args)...);
        // case 16:
        //     return dispatch_new<uint16_t, Op>(out, a, std::forward<Args>(args)...);
        case 32:
            return dispatch_new<uint32_t, Op>(out, a, std::forward<Args>(args)...);
        // case 64:
        //     return dispatch_new<uint64_t, Op>(out, a, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported uint bitwidth");
        }
    }
    default:
        throw std::runtime_error("Unsupported datatype code");
    }
}



/**
 * Deduce the first input datatype to a reduction op
*/
/**
 * out = f(a) where a is an ndarray and returns an ndarray
 * 
 * This requires two type deductions and passes through an underlying impl function with 2 template args
 * 
 * Used by
 * * mean
 * * stddev
 * * standardized_moment
 * * sum
 **/
template <Constraints constraints, typename Out, class Op, typename... Args>
ndarray constrained_dispatch_nd(ndarray &out, const ndarray &a, Args... args) {
    auto dtype = a.dtype();

    if (dtype.code == kDLFloat) {
        if constexpr(!has_constraint<constraints>(Constraints::NoFloat)) {
            switch (dtype.bits) {
            case 32:
                return Op::template call<Out, float>(out, a, std::forward<Args>(args)...);
            // case 64:
            //     return Op::template call<Out, double>(out, a, std::forward<Args>(args)...);
            default:
                throw std::runtime_error("Unsupported float bitwidth");
            }
        } else {
            throw std::runtime_error("Function does not support float dtypes");
        }
    } else if (dtype.code == kDLInt) {
        if constexpr(!has_constraint<constraints>(Constraints::NoInt)) {
            switch (dtype.bits) {
            // case 8:
            //     return Op::template call<Out, int8_t>(out, a, std::forward<Args>(args)...);
            // case 16:
            //     return Op::template call<Out, int16_t>(out, a, std::forward<Args>(args)...);
            case 32:
                return Op::template call<Out, int32_t>(out, a, std::forward<Args>(args)...);
            case 64:
                return Op::template call<Out, int64_t>(out, a, std::forward<Args>(args)...);
            default:
                throw std::runtime_error("Unsupported int bitwidth");
            }
        } else {
            throw std::runtime_error("Function does not support int dtypes");
        }
    } else if (dtype.code == kDLUInt) {
        if constexpr(!has_constraint<constraints>(Constraints::NoUInt)) {
            switch (dtype.bits) {
            case 8:
                return Op::template call<Out, uint8_t>(out, a, std::forward<Args>(args)...);
            // case 16:
            //     return Op::template call<Out, uint16_t>(out, a, std::forward<Args>(args)...);
            case 32:
                return Op::template call<Out, uint32_t>(out, a, std::forward<Args>(args)...);
            // case 64:
            //     return Op::template call<Out, uint64_t>(out, a, std::forward<Args>(args)...);
            default:
                throw std::runtime_error("Unsupported uint bitwidth");
            }
        } else {
            throw std::runtime_error("Function does not support unsigned int dtypes");
        }
    } else {
        throw std::runtime_error("Unsupported datatype code");
    }
}

/**
 * out = f(a) where out and a are ndarray
*/
// template <class Op, typename... Args>
// ndarray dispatch_new(ndarray &out, const ndarray &a, Args... args) {

template <Constraints constraints, class Op, typename... Args>
ndarray constrained_dispatch_nd(ndarray &out, const ndarray &a, Args... args) {
    auto dtype = out.dtype();

    if (dtype.code == kDLFloat) {
        if constexpr(!has_constraint<constraints>(Constraints::NoFloat)) {
            switch (dtype.bits) {
            case 32:
                return constrained_dispatch_nd<constraints, float, Op>(out, a, std::forward<Args>(args)...);
            // case 64:
            //     return dispatch_new<double, Op>(out, a, std::forward<Args>(args)...);
            default:
                throw std::runtime_error("Unsupported float bitwidth");
            }
        } else {
            throw std::runtime_error("Function does not support float dtypes");
        }
    } else if (dtype.code == kDLInt) {
        if constexpr(!has_constraint<constraints>(Constraints::NoInt)) {
            switch (dtype.bits) {
            // case 8:
            //     return dispatch_new<int8_t, Op>(constraints, out, a, std::forward<Args>(args)...);
            // case 16:
            //     return dispatch_new<int16_t, Op>(constraints, out, a, std::forward<Args>(args)...);
            case 32:
                return constrained_dispatch_nd<constraints, int32_t, Op>(out, a, std::forward<Args>(args)...);
            case 64:
                return constrained_dispatch_nd<constraints, int64_t, Op>(out, a, std::forward<Args>(args)...);
            default:
                throw std::runtime_error("Unsupported int bitwidth");
            }
        } else {
            throw std::runtime_error("Function does not support int dtypes");
        }
    } else if (dtype.code == kDLUInt) {
        if constexpr(!has_constraint<constraints>(Constraints::NoUInt)) {
            switch (dtype.bits) {
            case 8:
                return constrained_dispatch_nd<constraints, uint8_t, Op>(out, a, std::forward<Args>(args)...);
            // case 16:
            //     return dispatch_new<uint16_t, Op>(out, a, std::forward<Args>(args)...);
            case 32:
                return constrained_dispatch_nd<constraints, uint32_t, Op>(out, a, std::forward<Args>(args)...);
            // case 64:
            //     return dispatch_new<uint64_t, Op>(out, a, std::forward<Args>(args)...);
            default:
                throw std::runtime_error("Unsupported uint bitwidth");
            }
        } else {
            throw std::runtime_error("Function does not support unsigned int dtypes");
        }
    } else {
        throw std::runtime_error("Unsupported datatype code");
    }
}



/**
 * out = f(a) where a is ndarray, out = int64_t
 * 
 * used by
 * * count
*/
template <class Op, typename... Args>
int64_t dispatch_summary(const ndarray &a, Args... args) {
    auto dtype = a.dtype();

    switch (dtype.code) {

    case kDLFloat: {
        switch (dtype.bits) {
        case 32:
            return Op::template call<float>(a, std::forward<Args>(args)...);
        // case 64:
        //     return Op::template call<double>(a, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported float bitwidth");
        }
    }
    case kDLInt: {
        switch (dtype.bits) {
        // case 8:
        //     return Op::template call<int8_t>(a, std::forward<Args>(args)...);
        // case 16:
        //     return Op::template call<int16_t>(a, std::forward<Args>(args)...);
        case 32:
            return Op::template call<int32_t>(a, std::forward<Args>(args)...);
        // case 64:
        //     return Op::template call<int64_t>(a, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported int bitwidth");
        }
    }
    case kDLUInt: {
        switch (dtype.bits) {
        case 8:
            return Op::template call<uint8_t>(a, std::forward<Args>(args)...);
        // case 16:
        //     return Op::template call<uint16_t>(a, std::forward<Args>(args)...);
        case 32:
            return Op::template call<uint32_t>(a, std::forward<Args>(args)...);
        // case 64:
        //     return Op::template call<uint64_t>(a, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported uint bitwidth");
        }
    }
    default:
        throw std::runtime_error("Unsupported datatype code");
    }
}


/**
 * Deduce the datatype of a unary tensor operation
*/
template <typename F, typename Out, template <typename, typename> class Op, typename ...Args>
ndarray dispatch_new2(ndarray &out, const ndarray &a, Args... args) {
    auto a_dtype = a.dtype();

    switch (a_dtype.code) {

    case kDLFloat: {
        switch (a_dtype.bits) {
        case 32:
            return F::template call<Out, float, Op>(out, a, std::forward<Args>(args)...);
        // case 64:
        //     return F::template call<Out, double, Op>(out, a, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported float bitwidth");
        }
    }
    case kDLInt: {
        switch (a_dtype.bits) {
        // case 8:
        //     return F::template call<Out, int8_t, Op>(out, a, std::forward<Args>(args)...);
        // case 16:
        //     return F::template call<Out, int16_t, Op>(out, a, std::forward<Args>(args)...);
        case 32:
            return F::template call<Out, int32_t, Op>(out, a, std::forward<Args>(args)...);
        case 64:
            return F::template call<Out, int64_t, Op>(out, a, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported int bitwidth");
        }
    }
    case kDLUInt: {
        switch (a_dtype.bits) {
        case 8:
            return F::template call<Out, uint8_t, Op>(out, a, std::forward<Args>(args)...);
        // case 16:
        //     return F::template call<Out, uint16_t, Op>(out, a, std::forward<Args>(args)...);
        case 32:
            return F::template call<Out, uint32_t, Op>(out, a, std::forward<Args>(args)...);
        // case 64:
        //     return F::template call<Out, uint64_t, Op>(out, a, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported uint bitwidth");
        }
    }
    default:
        throw std::runtime_error("Unsupported datatype code");
    }
}



/**
 * Deduce the datatype of a unary tensor operation
*/
template <typename F, template <typename, typename> class Op, typename ...Args>
ndarray dispatch_new2(ndarray &out, const ndarray &a, Args... args) {
    auto dtype = out.dtype();

    switch (dtype.code) {

    case kDLFloat: {
        switch (dtype.bits) {
        case 32:
            return dispatch_new2<F, float, Op>(out, a, std::forward<Args>(args)...);
        // case 64:
        //     return dispatch_new2<F, double, Op>(out, a, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported float bitwidth");
        }
    }
    case kDLInt: {
        switch (dtype.bits) {
        // case 8:
        //     return dispatch_new2<F, int8_t, Op>(out, a, std::forward<Args>(args)...);
        // case 16:
        //     return dispatch_new2<F, int16_t, Op>(out, a, std::forward<Args>(args)...);
        case 32:
            return dispatch_new2<F, int32_t, Op>(out, a, std::forward<Args>(args)...);
        case 64:
            return dispatch_new2<F, int64_t, Op>(out, a, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported int bitwidth");
        }
    }
    case kDLUInt: {
        switch (dtype.bits) {
        case 8:
            return dispatch_new2<F, uint8_t, Op>(out, a, std::forward<Args>(args)...);
        // case 16:
        //     return dispatch_new2<F, uint16_t, Op>(out, a, std::forward<Args>(args)...);
        case 32:
            return dispatch_new2<F, uint32_t, Op>(out, a, std::forward<Args>(args)...);
        // case 64:
        //     return dispatch_new2<F, uint64_t, Op>(out, a, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported uint bitwidth");
        }
    }
    default:
        throw std::runtime_error("Unsupported datatype code");
    }
}



} // namespace bland
