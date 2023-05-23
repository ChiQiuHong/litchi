#pragma once

#include "litchi/core/params/params.h"

namespace litchi {

namespace core {

class fully_params : public Params {
public:
  size_t in_size_;
  size_t out_size_;
  bool has_bias_;
};

// TODO: can we do better here?
inline fully_params &Params::fully() {
  return *(static_cast<fully_params *>(this));
}

}  // namespace core

}  // namespace litchi