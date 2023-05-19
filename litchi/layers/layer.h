#pragma once

#include "litchi/node.h"

namespace litchi {

/**
 * base class of all kind of NN layers
 *
 * sub-class should override these methods:
 * - forward_propagation ... body of forward-pass calculation
 */
class layer : public node {
public:
  virtual ~layer() = default;

  /**
   * @brief Default layer constructor that instantiates a N-input, M-output
   * layer
   */
  layer(const size_t &in, const size_t &out) : node(in, out) {}
};

} // namespace litchi