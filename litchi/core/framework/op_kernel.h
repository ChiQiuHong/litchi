#pragma once

#include <memory>
#include <vector>

#include "litchi/core/params/params.h"

namespace litchi {

namespace core {

class OpKernel;  // delared below

class OpKernelConstruction {
 public:
  OpKernelConstruction() {}
  explicit OpKernelConstruction(Params *params) : params_(params) {}

  // Returns the params raw pointer
  Params *params() const { return params_; }

 private:
  Params *params_ = nullptr;
};

class OpKernelContext {
 public:
  struct OpParams {
    // the op kernel being computed.
    OpKernel *op_kernel_ptr = nullptr;

    // the layer on which kernel is running.
    layer *layer_ptr_ = nullptr;

    // the operation params
    Params *params_ptr_ = nullptr;

    backend_t engine = default_engine();
  };

  OpKernelContext()
    : in_data_(nullptr),
      out_data_(nullptr),
      out_grad_(nullptr),
      in_grad_(nullptr) {
    op_params_ = std::unique_ptr<OpParams>(new OpParams());
  }

  void set_in_out(const std::vector<tensor_t *> &in_data,
                  std::vector<tensor_t *> &out_data) {
    in_data_  = const_cast<std::vector<tensor_t *> *>(&in_data);
    out_data_ = const_cast<std::vector<tensor_t *> *>(&out_data);
  }

  tensor_t &input(const int idx) { return *(*in_data_)[idx]; }

  tensor_t &output(const int idx) { return *(*out_data_)[idx]; }

  backend_t engine() const { return op_params_->engine; }

  void setEngine(const backend_t engine) { op_params_->engine = engine; }

 private:
  std::vector<tensor_t *> *in_data_;
  std::vector<tensor_t *> *out_data_;
  std::vector<tensor_t *> *out_grad_;
  std::vector<tensor_t *> *in_grad_;

  std::unique_ptr<OpParams> op_params_;
};

class OpKernel {
 public:
  OpKernel() {}
  explicit OpKernel(const OpKernelConstruction &context)
    : params_(context.params()) {}

  virtual ~OpKernel() {}

  virtual void compute(OpKernelContext &context) = 0;

 protected:
  Params *params_ = nullptr;
};

}  // namespace core

}  // namespace litchi