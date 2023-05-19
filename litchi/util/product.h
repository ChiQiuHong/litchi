#pragma once

#include <cstdint>

#include "litchi/util/macro.h"

namespace vectorize {

namespace detail {

template <typename T>
void fill(T *dst, size_t size, T value) {
  std::fill(dst, dst + size, value);
}

} // namespace detail

template <typename T>
CNN_MUST_INLINE void fill(T *dst, std::size_t size, T value) {
  detail::fill(dst, size, value);
}

} // namespace vectorize