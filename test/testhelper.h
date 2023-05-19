#pragma once

#include <utility>

#include "litchi/litchi.h"
#include "litchi/util/gradient_check.h"

namespace litchi {

template <typename T> inline T epsilon() { return 0; }

template <> inline float epsilon() { return 1e-2f; }

template <> inline double epsilon() { return 1e-4; }

std::vector<tensor_t> generate_test_data(const std::vector<size_t> nsamples,
                                         const std::vector<size_t> sizes) {
  assert(nsamples.size() == sizes.size());
  std::vector<tensor_t> ret(nsamples.size());
  for (size_t i = 0; i < nsamples.size(); i++) {
    // for each tensor_t
    ret[i].resize(nsamples[i]);
    for (size_t j = 0; j < nsamples[i]; j++) {
      // for each vec_t
      ret[i][j].resize(sizes[i]);
      uniform_rand(ret[i][j].begin(), ret[i][j].end(), -1.0, 1.0);
    }
  }

  return ret;
}

} // namespace litchi