#pragma once

#include <memory>
#include <vector>

#include "litchi/layers/layer.h"

// #include "litchi/core/kernels/fully_connected_grad_op.h"
#include "litchi/core/kernels/fully_connected_op.h"

namespace litchi {

/**
 * compute fully-connected(matmul) operation
 */
class fully_connected_layer : public layer {
 public:
  /**
   * @param in_dim [in] number of elements of the input
   * @param out_dim [in] number of elements of the output
   * @param has_bias [in] whether to include additional bias to the layer
   */
  fully_connected_layer(size_t in_dim,
                        size_t out_dim,
                        bool has_bias                = true,
                        core::backend_t backend_type = core::default_engine())
    : layer(std_input_order(has_bias), {vector_type::data}) {
    set_params(in_dim, out_dim, has_bias);
    init_backend(backend_type);
    layer::set_backend_type(backend_type);
  }

  std::vector<index3d<size_t>> in_shape() const override {
    if (params_.has_bias_) {
      return {index3d<size_t>(params_.in_size_, 1, 1),
              index3d<size_t>(params_.in_size_, params_.out_size_, 1),
              index3d<size_t>(params_.out_size_, 1, 1)};
    } else {
      return {index3d<size_t>(params_.in_size_, 1, 1),
              index3d<size_t>(params_.in_size_, params_.out_size_, 1)};
    }
  }

  std::vector<index3d<size_t>> out_shape() const override {
    return {index3d<size_t>(params_.out_size_, 1, 1)};
  }

  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) override {
    // forward fully connected op context
    fwd_ctx_.set_in_out(in_data, out_data);
    fwd_ctx_.setEngine(layer::engine());

    // launch fully connected kernel
    kernel_fwd_->compute(fwd_ctx_);
  }

  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {}

 protected:
  void set_params(const size_t in_size, const size_t out_size, bool has_bias) {
    params_.in_size_  = in_size;
    params_.out_size_ = out_size;
    params_.has_bias_ = has_bias;
  }

  void init_backend(core::backend_t backend_type) {
    core::OpKernelConstruction ctx = core::OpKernelConstruction(&params_);

    if (backend_type == core::backend_t::internal) {
      kernel_fwd_.reset(new FullyConnectedOp(ctx));
    } else {
      // TODO error throw
      throw "Not supported engine: ";
    }
  }

 private:
  /* The layer parameters */
  core::fully_params params_;

  /* forward op context */
  core::OpKernelContext fwd_ctx_;

  /* backward op context */
  // core::OpKernelContext bwd_ctx_;

  /* Forward and backward ops */
  std::shared_ptr<core::OpKernel> kernel_fwd_;
  // std::shared_ptr<core::OpKernel> kernel_back_;
};

}  // namespace litchi