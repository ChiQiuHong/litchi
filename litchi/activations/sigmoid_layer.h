#pragma once

#include "litchi/activations/activation_layer.h"
#include "litchi/layers/layer.h"

namespace litchi {

class sigmoid_layer : public activation_layer {
 public:
  using activation_layer::activation_layer;

  void forward_activation(const vec_t &x, vec_t &y) override {
    for (size_t j = 0; j < x.size(); j++) {
      y[j] = float_t(1) / (float_t(1) + std::exp(-x[j]));
    }
  }

  void backward_activation(const vec_t &x,
                           const vec_t &y,
                           vec_t &dx,
                           const vec_t &dy) override {
    for (size_t j = 0; j < x.size(); j++) {
      // dx = dy * (gradient of sigmoid)
      dx[j] = dy[j] * y[j] * (float_t(1) - y[j]);
    }
  }
};

}  // namespace litchi