#include <preprocess/passband_static_equalize.hpp>
#include <preprocess/normalize.hpp>
#include <preprocess/excise_dc.hpp>
#include <holoscan/holoscan.hpp>

namespace bliss::holoscan_ops {
    class PassbandEqualizeOp : public holoscan::Operator {
        public:
        HOLOSCAN_OPERATOR_FORWARD_ARGS(PassbandEqualizeOp)
        PassbandEqualizeOp(std::string h_resp_filepath) {
            _equalizer_path = h_resp_filepath;
            _equalizer_taps = bland::read_from_file(_equalizer_path, bland::ndarray::datatype::float32);
        }

        void setup(holoscan::OperatorSpec& spec) override {
            spec.input<bliss::coarse_channel>("input");
            spec.output<bliss::coarse_channel>("output");
        }

        void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
                    holoscan::ExecutionContext& context) override {
            auto input_cc = op_input.receive<bliss::coarse_channel>("input").value();
            auto equalized_cc = bliss::equalize_passband_filter(input_cc, _equalizer_taps);
            op_output.emit(equalized_cc, "output");
        }

        private:
        std::string _equalizer_path;
        bland::ndarray _equalizer_taps;
    };


    class NormalizeOp : public holoscan::Operator {
        public:
        HOLOSCAN_OPERATOR_FORWARD_ARGS(NormalizeOp)
        NormalizeOp() = default;

        void setup(holoscan::OperatorSpec& spec) override {
            spec.input<bliss::coarse_channel>("input");
            spec.output<bliss::coarse_channel>("output");
        }

        void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
                    holoscan::ExecutionContext& context) override {
            auto input_cc = op_input.receive<bliss::coarse_channel>("input").value();
            auto normalized_cc = bliss::normalize(input_cc);
            op_output.emit(normalized_cc, "output");
        }

        private:
    };

    class ExciseDCOp : public holoscan::Operator {
        public:
        HOLOSCAN_OPERATOR_FORWARD_ARGS(ExciseDCOp)
        ExciseDCOp() = default;

        void setup(holoscan::OperatorSpec& spec) override {
            spec.input<bliss::coarse_channel>("input");
            spec.output<bliss::coarse_channel>("output");
        }

        void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
                    holoscan::ExecutionContext& context) override {
            auto input_cc = op_input.receive<bliss::coarse_channel>("input").value();
            auto normalized_cc = bliss::excise_dc(input_cc);
            op_output.emit(normalized_cc, "output");
        }

        private:
    };
}