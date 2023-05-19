#pragma once

#include "litchi/activations/activation_layer.h"
#include "litchi/layers/layer.h"

namespace litchi {

class relu_layer : public activation_layer {
public:
  using activation_layer::activation_layer;

  void forward_activation(const vec_t &x, vec_t &y) override {
    for (size_t j = 0; j < x.size(); j++) {
      y[j] = std::max(float_t(0), x[j]);
    }
  }

  void backward_activation(const vec_t &x, const vec_t &y, vec_t &dx,
                           const vec_t &dy) override {
    for (size_t j = 0; j < x.size(); j++) {
      // dx = dy * (gradient of relu)
      dx[j] = dy[j] * (y[j] > float(0) ? float(1) : float(0));
    }
  }
};

} // namespace litchi