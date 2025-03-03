#include <preprocess/passband_static_equalize.hpp>
#include <holoscan/holoscan.hpp>

namespace bliss::holoscan_ops {
    class PassbandEqualizeOp : public holoscan::Operator {
        public:
        HOLOSCAN_OPERATOR_FORWARD_ARGS(HelloWorldOp)
        PassbandEqualizeOp(std::string h_resp_filepath) = default;
        void setup(holoscan::OperatorSpec& spec) override {
            auto _equalizer_taps = bland::read_from_file(h_resp_filepath, bland::ndarray::datatype::float32);
            spec.input("input", bliss::coarse_channel, {1});
            spec.output("output", bliss::coarse_channel, {1});
        }

        void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
                    [[maybe_unused]] ExecutionContext& context) override {
            auto input_cc = op_input.get<bliss::coarse_channel>("input");
            auto equalized_cc = bliss::equalize_passband_filter(input_cc, _equalizer_taps);
            op_output.emit("output", equalized_cc);
        }

        private:
        std::string _equalizer_path;
        bland::ndarray _equalizer_taps;
    };
}