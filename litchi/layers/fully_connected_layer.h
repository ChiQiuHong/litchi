#pragma once

#include <vector>

#include "litchi/layers/layer.h"

namespace litchi {

/**
 * compute fully-connected(matmul) operation
 */
// class fully_connected_layer : public layer {
// public:
//   /**
//    * @param in_dim [in] number of elements of the input
//    * @param out_dim [in] number of elements of the output
//    * @param has_bias [in] whether to include additional bias to the layer
//    */
//   fully_connected_layer(size_t in_dim, size_t out_dim, bool has_bias)
//       : layer({vector_type::data}, {vector_type::data}) {}

//   std::vector<index3d<size_t>> in_shape() const override {
//     if (params_.has_bias_) {
//       return {index3d<size_t>(params_.in_size_, 1, 1),
//               index3d<size_t>(params_.in_size_, params_.out_size_, 1),
//               index3d<size_t>(params_.out_size_, 1, 1)};
//     } else {
//       return {index3d<size_t>(params_.in_size_, 1, 1),
//               index3d<size_t>(params_.in_size_, params_.out_size_, 1)};
//     }
//   }

//   std::vector<index3d<size_t>> out_shape() const override {
//     return {index3d<size_t>(params_.out_size_, 1, 1)};
//   }
// };

} // namespace litchi