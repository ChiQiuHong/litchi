#pragma once

#include "litchi/activations/activation_layer.h"
#include "litchi/layers/layer.h"

namespace litchi {

class relu_layer : public activation_layer {
public:
  using activation_layer::activation_layer;

  void forward_activation(const std::vector<float> &x, std::vector<float> &y) {
    for (size_t j = 0; j < x.size(); j++) {
      y[j] = std::max(float(0), x[j]);
    }
  }
};

} // namespace litchi