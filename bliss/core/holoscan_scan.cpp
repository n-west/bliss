#include <core/scan.hpp>
#include <holoscan/holoscan.hpp>

namespace bliss::holoscan_ops {
class ScanFilterbankSourceOp : public holoscan::Operator {
  public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(ScanFilterbankSourceOp)
    ScanFilterbankSourceOp(std::string filterbank_h5_path, int num_fine_channels_per_coarse = 0) :
            _filterbank_h5_path(filterbank_h5_path),
            _num_fine_channels_per_coarse(num_fine_channels_per_coarse) {};

    void setup(holoscan::OperatorSpec &spec) override {
        _scan = bliss::scan(_filterbank_h5_path, _num_fine_channels_per_coarse);
        spec.output<bliss::coarse_channel>("output");
    }

    void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
                holoscan::ExecutionContext& context) override {
        auto cc = _scan.read_coarse_channel(_coarse_channel_index);
        op_output.emit(cc, "output");
    }

  private:
    std::string _filterbank_h5_path;
    int         _num_fine_channels_per_coarse;
    int        _coarse_channel_index = 0;
    bliss::scan _scan;
};
} // namespace bliss::holoscan_ops