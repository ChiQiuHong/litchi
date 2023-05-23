#pragma once

#include "litchi/core/params/fully_params.h"

namespace litchi {

namespace kernels {

inline void fully_connected_op_internal(const tensor_t &in_data,
                                        const vec_t &W,
                                        const vec_t &bias,
                                        tensor_t &out_data,
                                        const core::fully_params &params) {
  // TODO parallel for
  for (size_t sample = 0; sample < in_data.size(); ++sample) {
    const vec_t &in = in_data[sample];
    vec_t &out      = out_data[sample];

    for (size_t i = 0; i < params.out_size_; i++) {
      out[i] = float_t{0};
      for (size_t c = 0; c < params.in_size_; c++) {
        out[i] += W[c * params.out_size_ + i] * in[c];
      }

      if (params.has_bias_) {
        out[i] += bias[i];
      }
    }
  }
}

}  // namespace kernels

}  // namespace litchi