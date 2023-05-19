#pragma once

#include <cstddef>
#include <vector>

namespace litchi {

/**
 * base class of all kind of data
 */
class node {
public:
  node(size_t in_size, size_t out_size) {}
  virtual ~node() {}

  std::vector<size_t> prev_;
  std::vector<size_t> next_;
};

} // namespace litchi