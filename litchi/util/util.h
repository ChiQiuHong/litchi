#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#include "litchi/util/macro.h"
#include "litchi/util/product.h"
#include "litchi/util/random.h"

namespace litchi {

typedef std::vector<float_t> vec_t;

typedef std::vector<vec_t> tensor_t;

template <typename T>
struct index3d {
  index3d(T width, T height, T depth) { reshape(width, height, depth); }

  index3d() : width_(0), height_(0), depth_(0) {}

  void reshape(T width, T height, T depth) {
    width_  = width;
    height_ = height;
    depth_  = depth;

    // TODO throw something
    // if ((int64_t)width * height * depth > std::numeric_limits<T>::max())
  }

  T get_index(T x, T y, T channel) const {
    assert(x >= 0 && x < width_);
    assert(y >= 0 && y < height_);
    assert(channel >= 0 && channel < depth_);
    return (height_ * channel + y) * width_ + x;
  }

  T area() const { return width_ * height_; }

  T size() const { return width_ * height_ * depth_; }

  T width_;
  T height_;
  T depth_;
};

typedef index3d<size_t> shape3d;

enum class vector_type : int32_t {
  // 0x0001xxx : in/out data
  data = 0x0001000, // input/output data, fed by other layer or input channel

  // 0x0002xxx : trainable parameters, updated for each back propagation
  weight = 0x0002000,
  bias   = 0x0002001,

  label = 0x0004000,
  aux   = 0x0010000 // layer-specific storage
};

inline void fill_tensor(tensor_t &tensor, float_t value) {
  for (auto &t : tensor) {
    vectorize::fill(&t[0], t.size(), value);
  }
}

inline void fill_tensor(tensor_t &tensor, float_t value, size_t size) {
  for (auto &t : tensor) {
    t.resize(size, value);
  }
}

} // namespace litchi