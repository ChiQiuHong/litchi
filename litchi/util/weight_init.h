#pragma once

#include "litchi/util/util.h"

namespace litchi {

namespace weight_init {

class function {
 public:
  virtual void fill(vec_t *weight, size_t fan_in, size_t fan_out) = 0;
};

class scalable : public function {
 public:
  explicit scalable(float_t value) : scale_(value) {}

  void scale(float_t value) { scale_ = value; }

 protected:
  float_t scale_;
};

/**
 * Use fan-in and fan-out for scaling
 *
 * X Glorot, Y Bengio,
 * Understanding the difficulty of training deep feedforward neural networks
 * Proc. AISTATS 10, May 2010, vol.9, pp249-256
 **/
class xavier : public scalable {
 public:
  xavier() : scalable(float_t(6)) {}
  explicit xavier(float_t value) : scalable(value) {}

  void fill(vec_t *weight, size_t fan_in, size_t fan_out) override {
    const float_t weight_base = std::sqrt(scale_ / (fan_in + fan_out));

    uniform_rand(weight->begin(), weight->end(), -weight_base, weight_base);
  }
};

class constant : public scalable {
 public:
  constant() : scalable(float_t{0}) {}
  explicit constant(float_t value) : scalable(value) {}

  void fill(vec_t *weight, size_t fan_in, size_t fan_out) override {
    CNN_UNREFERENCED_PARAMETER(fan_in);
    CNN_UNREFERENCED_PARAMETER(fan_out);
    vectorize::fill(&(*weight)[0], weight->size(), scale_);
  }
};

}  // namespace weight_init

}  // namespace litchi