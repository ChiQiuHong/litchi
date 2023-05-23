#pragma once

#include "litchi/core/framework/op_kernel.h"

#include "litchi/core/kernels/fully_connected_op_internal.h"

namespace litchi {

class FullyConnectedOp : public core::OpKernel {
 public:
  explicit FullyConnectedOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    auto params = OpKernel::params_->fully();

    // incoming/outcoming data
    const tensor_t &in_data = context.input(0);
    const tensor_t &W       = context.input(1);
    const tensor_t *bias    = params.has_bias_ ? &context.input(2) : nullptr;
    tensor_t &out_data      = context.output(0);

    // initialize outputs
    fill_tensor(out_data, float_t{0});

    // call the algorithm depending on the selected engine type
    const core::backend_t engine = context.engine();

    if (engine == core::backend_t::internal) {
      kernels::fully_connected_op_internal(
        in_data, W[0], params.has_bias_ ? (*bias)[0] : vec_t(), out_data,
        params);
    } else {
      throw "Not supported engine";
    }
  }
};

}  // namespace litchi