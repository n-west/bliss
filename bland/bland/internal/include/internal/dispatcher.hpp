#pragma once

#include "bland/ndarray.hpp"

#include <fmt/core.h>

#include <stdexcept>

namespace bland {

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
 * out = f(a, b) where a is ndarray and b is a scalar that outputs an ndarray
 * 
 * This requires three type deductions and passes through an underlying impl function with 3 template args
 **/
template <typename F, typename Out, typename S, class Op, typename ...Args>
ndarray dispatch_new3(ndarray &out, const ndarray &a, const S &b, Args... args) {
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
ndarray dispatch_new3(ndarray &out, const ndarray &a, const S &b, Args... args) {
    auto dtype = out.dtype();

    switch (dtype.code) {

    case kDLFloat: {
        switch (dtype.bits) {
        case 32:
            return dispatch_new3<F, float, S, Op>(out, a, b, std::forward<Args>(args)...);
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
            return dispatch_new3<F, int32_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        case 64:
            return dispatch_new3<F, int64_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported int bitwidth");
        }
    }
    case kDLUInt: {
        switch (dtype.bits) {
        case 8:
            return dispatch_new3<F, uint8_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        // case 16:
        //     return dispatch_new3<F, uint16_t, S, Op>(out, a, b, std::forward<Args>(args)...);
        case 32:
            return dispatch_new3<F, uint32_t, S, Op>(out, a, b, std::forward<Args>(args)...);
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
 * out = f(a) where a and c are ndarrays.
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