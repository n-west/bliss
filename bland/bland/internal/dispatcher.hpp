#pragma once

#include "bland/ndarray.hpp"

#include <stdexcept>

namespace bland {

/**
 * Dispatch operations to implementations templated on actual datatypes
 * c = op(a, b)
 * by reading the ndarray dtype and filling in appropriate template arguments.
 * 
 * This is highly verbose, but conceptually simple even though the syntax is somewhat advanced
 * to get datatypes, template substitution, and perfect argument forwarding
 *
 * // TODO: add device dispatching too!
 */

/**
 * Dispatch a two-argument function where the underlying datatype is already known
*/
template <typename F, typename A_type, template <typename, typename> class Op, typename ...Args>
ndarray dispatch(const ndarray &a, const ndarray &b, Args... args) {
    auto b_dtype = b.dtype();

    switch (b_dtype.code) {
    case kDLFloat:
        switch (b_dtype.bits) {
        case 32:
            return F::template call<A_type, float, Op>(a, b, std::forward<Args>(args)...);
        case 64:
            return F::template call<A_type, double, Op>(a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported bitwidth for float");
        }
    case kDLInt:
        switch (b_dtype.bits) {
        case 8:
            return F::template call<A_type, int8_t, Op>(a, b, std::forward<Args>(args)...);
        case 16:
            return F::template call<A_type, int16_t, Op>(a, b, std::forward<Args>(args)...);
        case 32:
            return F::template call<A_type, int32_t, Op>(a, b, std::forward<Args>(args)...);
        case 64:
            return F::template call<A_type, int64_t, Op>(a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported bitwidth for int");
        }
    case kDLUInt:
        switch (b_dtype.bits) {
        case 8:
            return F::template call<A_type, uint8_t, Op>(a, b, std::forward<Args>(args)...);
        case 16:
            return F::template call<A_type, uint16_t, Op>(a, b, std::forward<Args>(args)...);
        case 32:
            return F::template call<A_type, uint32_t, Op>(a, b, std::forward<Args>(args)...);
        case 64:
            return F::template call<A_type, uint64_t, Op>(a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported bitwidth for int");
        }

    default:
        throw std::runtime_error("Unsupported dtype");
    }
}

/**
 * Deduce the datatype of first argument of a two-argument function and template substitute
 * to deduce the second argument
*/
template <typename F, template <typename, typename> class Op, typename ...Args>
ndarray dispatch(const ndarray &a, const ndarray &b, Args... args) {
    auto a_dtype = a.dtype();

    switch (a_dtype.code) {
    case kDLFloat: {
        switch (a_dtype.bits) {
        case 32:
            return dispatch<F, float, Op>(a, b, std::forward<Args>(args)...);
        case 64:
            return dispatch<F, double, Op>(a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported float bitwidth");
        }
    }
    case kDLInt: {
        switch (a_dtype.bits) {
        case 8:
            return dispatch<F, int8_t, Op>(a, b, std::forward<Args>(args)...);
        case 16:
            return dispatch<F, int16_t, Op>(a, b, std::forward<Args>(args)...);
        case 32:
            return dispatch<F, int32_t, Op>(a, b, std::forward<Args>(args)...);
        case 64:
            return dispatch<F, int64_t, Op>(a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported int bitwidth");
        }
    }
    case kDLUInt: {
        switch (a_dtype.bits) {
        case 8:
            return dispatch<F, uint8_t, Op>(a, b, std::forward<Args>(args)...);
        case 16:
            return dispatch<F, uint16_t, Op>(a, b, std::forward<Args>(args)...);
        case 32:
            return dispatch<F, uint32_t, Op>(a, b, std::forward<Args>(args)...);
        case 64:
            return dispatch<F, uint64_t, Op>(a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported uint bitwidth");
        }
    }
    default:
        throw std::runtime_error("Unsupported datatype code");
    }
}

/**
 * Deduce the datatype of a tensor operating with a scalar
*/
template <typename F, typename S, template <typename, typename> class Op, typename ...Args>
ndarray dispatch(const ndarray &a, const S &b, Args... args) {
    auto a_dtype = a.dtype();

    switch (a_dtype.code) {

    case kDLFloat: {
        switch (a_dtype.bits) {
        case 32:
            return F::template call<float, S, Op>(a, b, std::forward<Args>(args)...);
        case 64:
            return F::template call<double, S, Op>(a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported float bitwidth");
        }
    }
    case kDLInt: {
        switch (a_dtype.bits) {
        case 8:
            return F::template call<int8_t, S, Op>(a, b, std::forward<Args>(args)...);
        case 16:
            return F::template call<int16_t, S, Op>(a, b, std::forward<Args>(args)...);
        case 32:
            return F::template call<int32_t, S, Op>(a, b, std::forward<Args>(args)...);
        case 64:
            return F::template call<int64_t, S, Op>(a, b, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported int bitwidth");
        }
    }
    case kDLUInt: {
        switch (a_dtype.bits) {
        case 8:
            return F::template call<uint8_t, S, Op>(a, b, std::forward<Args>(args)...);
        case 16:
            return F::template call<uint16_t, S, Op>(a, b, std::forward<Args>(args)...);
        case 32:
            return F::template call<uint32_t, S, Op>(a, b, std::forward<Args>(args)...);
        case 64:
            return F::template call<uint64_t, S, Op>(a, b, std::forward<Args>(args)...);
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
template <typename F, template <typename> class Op, typename ...Args>
ndarray dispatch(const ndarray &a, Args... args) {
    auto a_dtype = a.dtype();

    switch (a_dtype.code) {

    case kDLFloat: {
        switch (a_dtype.bits) {
        case 32:
            return F::template call<float, Op>(a, std::forward<Args>(args)...);
        case 64:
            return F::template call<double, Op>(a, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported float bitwidth");
        }
    }
    case kDLInt: {
        switch (a_dtype.bits) {
        case 8:
            return F::template call<int8_t, Op>(a, std::forward<Args>(args)...);
        case 16:
            return F::template call<int16_t, Op>(a, std::forward<Args>(args)...);
        case 32:
            return F::template call<int32_t, Op>(a, std::forward<Args>(args)...);
        case 64:
            return F::template call<int64_t, Op>(a, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported int bitwidth");
        }
    }
    case kDLUInt: {
        switch (a_dtype.bits) {
        case 8:
            return F::template call<uint8_t, Op>(a, std::forward<Args>(args)...);
        case 16:
            return F::template call<uint16_t, Op>(a, std::forward<Args>(args)...);
        case 32:
            return F::template call<uint32_t, Op>(a, std::forward<Args>(args)...);
        case 64:
            return F::template call<uint64_t, Op>(a, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported uint bitwidth");
        }
    }
    default:
        throw std::runtime_error("Unsupported datatype code");
    }
}


/**
 * 
 * *********
 * TODO: Below this line, the dispatchers all accept an "out" argument. Above this line they do not
 * 
 * (1) Deduce the "out" argument dtype as part of the template and trust that the calling op has created
 * dtypes that make sense for their specific operation
 * (2) Update all above this to accept an "out" argument
 * *********
 * 
*/

/**
 * Deduce the datatype of a unary tensor operation that stores result in provided tensor
*/
template <typename F, template <typename> class Op, typename ...Args>
ndarray dispatch(const ndarray &a, ndarray &out, Args... args) {
    auto a_dtype = a.dtype();

    switch (a_dtype.code) {

    case kDLFloat: {
        switch (a_dtype.bits) {
        case 32:
            return F::template call<float, Op>(a, out, std::forward<Args>(args)...);
        case 64:
            return F::template call<double, Op>(a, out, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported float bitwidth");
        }
    }
    case kDLInt: {
        switch (a_dtype.bits) {
        case 8:
            return F::template call<int8_t, Op>(a, out, std::forward<Args>(args)...);
        case 16:
            return F::template call<int16_t, Op>(a, out, std::forward<Args>(args)...);
        case 32:
            return F::template call<int32_t, Op>(a, out, std::forward<Args>(args)...);
        case 64:
            return F::template call<int64_t, Op>(a, out, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported int bitwidth");
        }
    }
    case kDLUInt: {
        switch (a_dtype.bits) {
        case 8:
            return F::template call<uint8_t, Op>(a, out, std::forward<Args>(args)...);
        case 16:
            return F::template call<uint16_t, Op>(a, out, std::forward<Args>(args)...);
        case 32:
            return F::template call<uint32_t, Op>(a, out, std::forward<Args>(args)...);
        case 64:
            return F::template call<uint64_t, Op>(a, out, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported uint bitwidth");
        }
    }
    default:
        throw std::runtime_error("Unsupported datatype code");
    }
}


/**
 * Deduce datatype of output for functions which take a scalar (think fill)
*/
template <typename F, typename S, typename ...Args>
ndarray dispatch(S a, ndarray &out, Args... args) {
    auto out_dtype = out.dtype();

    switch (out_dtype.code) {

    case kDLFloat: {
        switch (out_dtype.bits) {
        case 32:
            return F::template call<float>(a, out, std::forward<Args>(args)...);
        case 64:
            return F::template call<double>(a, out, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported float bitwidth");
        }
    }
    case kDLInt: {
        switch (out_dtype.bits) {
        case 8:
            return F::template call<int8_t>(a, out, std::forward<Args>(args)...);
        case 16:
            return F::template call<int16_t>(a, out, std::forward<Args>(args)...);
        case 32:
            return F::template call<int32_t>(a, out, std::forward<Args>(args)...);
        case 64:
            return F::template call<int64_t>(a, out, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported int bitwidth");
        }
    }
    case kDLUInt: {
        switch (out_dtype.bits) {
        case 8:
            return F::template call<uint8_t>(a, out, std::forward<Args>(args)...);
        case 16:
            return F::template call<uint16_t>(a, out, std::forward<Args>(args)...);
        case 32:
            return F::template call<uint32_t>(a, out, std::forward<Args>(args)...);
        case 64:
            return F::template call<uint64_t>(a, out, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported uint bitwidth");
        }
    }
    default:
        throw std::runtime_error("Unsupported datatype code");
    }
}


// /**
//  * Deduce datatype of output for functions which take a scalar argument *before* the input array (logic operators, but think through if we should have this or if it's simpler to just do it another way)
// */
// template <typename F, typename S, typename ...Args>
// ndarray dispatch(S a, ndarray a, ndarray &out, Args... args) {
//     auto out_dtype = out.dtype();

//     switch (out_dtype.code) {

//     case kDLFloat: {
//         switch (out_dtype.bits) {
//         case 32:
//             return F::template call<float>(a, out, std::forward<Args>(args)...);
//         case 64:
//             return F::template call<double>(a, out, std::forward<Args>(args)...);
//         default:
//             throw std::runtime_error("Unsupported float bitwidth");
//         }
//     }
//     case kDLInt: {
//         switch (out_dtype.bits) {
//         case 8:
//             return F::template call<int8_t>(a, out, std::forward<Args>(args)...);
//         case 16:
//             return F::template call<int16_t>(a, out, std::forward<Args>(args)...);
//         case 32:
//             return F::template call<int32_t>(a, out, std::forward<Args>(args)...);
//         case 64:
//             return F::template call<int64_t>(a, out, std::forward<Args>(args)...);
//         default:
//             throw std::runtime_error("Unsupported int bitwidth");
//         }
//     }
//     case kDLUInt: {
//         switch (out_dtype.bits) {
//         case 8:
//             return F::template call<uint8_t>(a, out, std::forward<Args>(args)...);
//         case 16:
//             return F::template call<uint16_t>(a, out, std::forward<Args>(args)...);
//         case 32:
//             return F::template call<uint32_t>(a, out, std::forward<Args>(args)...);
//         case 64:
//             return F::template call<uint64_t>(a, out, std::forward<Args>(args)...);
//         default:
//             throw std::runtime_error("Unsupported uint bitwidth");
//         }
//     }
//     default:
//         throw std::runtime_error("Unsupported datatype code");
//     }
// }


/*******
 * Above this line we do not deduce output dtype. Below it, we do
 * TODO: deduce output dtype on all
 *******
*/


/**
 * Deduce the output datatype to a reduction op, forwarding all arguments
*/
template <typename in_dtype, class Op, typename... Args>
ndarray dispatch(const ndarray &a, ndarray &out, std::vector<int64_t> axes, Args... args) {
    auto dtype = out.dtype();

    switch (dtype.code) {

    case kDLFloat: {
        switch (dtype.bits) {
        case 32:
            return Op::template call<in_dtype, float>(a, out, axes, std::forward<Args>(args)...);
        case 64:
            return Op::template call<in_dtype, double>(a, out, axes, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported float bitwidth");
        }
    }
    case kDLInt: {
        switch (dtype.bits) {
        case 8:
            return Op::template call<in_dtype, int8_t>(a, out, axes, std::forward<Args>(args)...);
        case 16:
            return Op::template call<in_dtype, int16_t>(a, out, axes, std::forward<Args>(args)...);
        case 32:
            return Op::template call<in_dtype, int32_t>(a, out, axes, std::forward<Args>(args)...);
        case 64:
            return Op::template call<in_dtype, int64_t>(a, out, axes, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported int bitwidth");
        }
    }
    case kDLUInt: {
        switch (dtype.bits) {
        case 8:
            return Op::template call<in_dtype, uint8_t>(a, out, axes, std::forward<Args>(args)...);
        case 16:
            return Op::template call<in_dtype, uint16_t>(a, out, axes, std::forward<Args>(args)...);
        case 32:
            return Op::template call<in_dtype, uint32_t>(a, out, axes, std::forward<Args>(args)...);
        case 64:
            return Op::template call<in_dtype, uint64_t>(a, out, axes, std::forward<Args>(args)...);
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
template <class Op, typename... Args>
ndarray dispatch(const ndarray &a, ndarray &out, std::vector<int64_t> axes, Args... args) {
    auto a_dtype = a.dtype();

    switch (a_dtype.code) {

    case kDLFloat: {
        switch (a_dtype.bits) {
        case 32:
            return dispatch<float, Op>(a, out, axes, std::forward<Args>(args)...);
        case 64:
            return dispatch<double, Op>(a, out, axes, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported float bitwidth");
        }
    }
    case kDLInt: {
        switch (a_dtype.bits) {
        case 8:
            return dispatch<int8_t, Op>(a, out, axes, std::forward<Args>(args)...);
        case 16:
            return dispatch<int16_t, Op>(a, out, axes, std::forward<Args>(args)...);
        case 32:
            return dispatch<int32_t, Op>(a, out, axes, std::forward<Args>(args)...);
        case 64:
            return dispatch<int64_t, Op>(a, out, axes, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported int bitwidth");
        }
    }
    case kDLUInt: {
        switch (a_dtype.bits) {
        case 8:
            return dispatch<uint8_t, Op>(a, out, axes, std::forward<Args>(args)...);
        case 16:
            return dispatch<uint16_t, Op>(a, out, axes, std::forward<Args>(args)...);
        case 32:
            return dispatch<uint32_t, Op>(a, out, axes, std::forward<Args>(args)...);
        case 64:
            return dispatch<uint64_t, Op>(a, out, axes, std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported uint bitwidth");
        }
    }
    default:
        throw std::runtime_error("Unsupported datatype code");
    }
}

} // namespace bland