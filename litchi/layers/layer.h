#pragma once

#include "litchi/node.h"

#include "litchi/util/util.h"

namespace litchi {

/**
 * base class of all kind of NN layers
 *
 * sub-class should override these methods:
 * - forward_propagation ... body of forward-pass calculation
 * - back_propagation    ... body of backward-pass calculation
 */
class layer : public node {
public:
  virtual ~layer() = default;

  /**
   * @brief Default layer constructor that instantiates a N-input, M-output
   * layer
   *
   * @param in_type[N] type of input vector (data, weight, bias...)
   * @param out_type[M] type of output vector
   */
  layer(const std::vector<vector_type> &in_type,
        const std::vector<vector_type> &out_type)
      : node(in_type.size(), out_type.size()), in_type_(in_type),
        out_type_(out_type) {}

  /**
   * array of input shapes (width x height x depth)
   */
  virtual std::vector<shape3d> in_shape() const = 0;

  /**
   * array of output shapes (width x height x depth)
   */
  virtual std::vector<shape3d> out_shape() const = 0;

  /////////////////////////////////////////////////////////////////////////
  // fprop/bprop

  /**
   * @param in_data input vectors of this layer (data, weight, bias)
   * @param out_data output vectors
   */
  virtual void forward_propagation(const std::vector<tensor_t *> &in_data,
                                   std::vector<tensor_t *> &out_data) = 0;

  /**
   * return delta of previous layer (delta=\frac{dE}{da}, a=wx in
   * fully-connected layer)
   * @param in_data  input vectors (same vectors as forward_propagation)
   * @param out_data output vectors (same vectors as forward_propagation)
   * @param out_grad gradient of output vectors (i-th vector correspond with
   * out_data[i])
   * @param in_grad  gradient of input vectors (i-th vector correspond with
   * in_data[i])
   */
  virtual void back_propagation(const std::vector<tensor_t *> &in_data,
                                const std::vector<tensor_t *> &out_data,
                                std::vector<tensor_t *> &out_grad,
                                std::vector<tensor_t *> &in_grad) = 0;

protected:
  /** Vector containing the type of data for inputs */
  std::vector<vector_type> in_type_;
  /** Vector containing the type of data for outputs */
  std::vector<vector_type> out_type_;
};

}  // namespace litchi