#pragma once

#include "litchi/core/params/fully_params.h"
#include "litchi/layers/layer.h"

namespace litchi {

namespace core {

enum class backend_t { internal };

inline backend_t default_engine() { return backend_t::internal; }

// class backend {
//  public:
//   // context holds solution-dependent parameters
//   // context should be able to hold any types of structures (like boost::any)
//   explicit backend(context *ctx_ = nullptr) {
//     CNN_UNREFERENCED_PARAMETER(ctx_);
//   }

//  protected:
//   context *ctx_;
//   layer *layer_;
// };

}  // namespace core

}  // namespace litchi