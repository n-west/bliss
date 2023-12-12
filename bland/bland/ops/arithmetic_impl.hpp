
namespace bland
{
    
/**
 * Template struct to allow passing an elementwise addition as a template argument
*/

template <typename Out, typename A, typename B>
struct elementwise_add_op_ts {
    /**
     * elementwise addition between two scalars in an ndarray with type `A` and `B`
    */
    static inline Out call(const A &a, const B &b) { return static_cast<Out>(a + b); }
};

/**
 * Template struct to allow passing an elementwise subtraction as a template argument
*/
template <typename Out, typename A, typename B>
struct elementwise_subtract_op_ts {
    /**
     * elementwise subtraction between two scalars in an ndarray with type `A` and `B`
    */
    static inline Out call(const A &a, const B &b) { return static_cast<Out>(a - b); }
};

/**
 * Template struct to allow passing an elementwise multiplication as a template argument
*/
template <typename Out, typename A, typename B>
struct elementwise_multiply_op_ts {
    /**
     * elementwise multiplication between two scalars in an ndarray with type `A` and `B`
    */
    static inline Out call(const A &a, const B &b) { return static_cast<Out>(a * b); }
};

/**
 * Template struct to allow passing an elementwise division as a template argument
*/
template <typename Out, typename A, typename B>
struct elementwise_divide_op_ts {
    /**
     * elementwise division between two scalars in an ndarray with type `A` and `B`
    */
    static inline Out call(const A &a, const B &b) { return static_cast<Out>(a / b); }
};

} // namespace bland
