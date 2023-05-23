#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "litchi/util/util.h"

namespace litchi {

class node;
class layer;
class edge;

typedef std::shared_ptr<edge> edgeptr_t;

/**
 * base class of all kind of data
 */
class node : public std::enable_shared_from_this<node> {
 public:
  node(size_t in_size, size_t out_size) : prev_(in_size), next_(out_size) {}
  virtual ~node() {}

  const std::vector<edgeptr_t> &prev() const { return prev_; }
  const std::vector<edgeptr_t> &next() const { return next_; }

 protected:
  node() = delete;

  mutable std::vector<edgeptr_t> prev_;
  mutable std::vector<edgeptr_t> next_;
};

/**
 * class containing input/output data
 */
class edge {
 public:
  edge(node *prev, const shape3d &shape, vector_type vtype)
    : shape_(shape),
      vtype_(vtype),
      data_({vec_t(shape.size())}),
      grad_({vec_t(shape.size())}),
      prev_(prev) {}

  void clear_grads() {
    for (size_t sample = 0, sample_count = grad_.size(); sample < sample_count;
         ++sample) {
      auto &g = grad_[sample];
      vectorize::fill(&g[0], g.size(), float_t{0});
    }
  }

  tensor_t *get_data() { return &data_; }

  const tensor_t *get_data() const { return &data_; }

  tensor_t *get_gradient() { return &grad_; }

  const tensor_t *get_gradient() const { return &grad_; }

  const shape3d &shape() const { return shape_; }

 private:
  shape3d shape_;
  vector_type vtype_;
  tensor_t data_;
  tensor_t grad_;
  node *prev_;
  std::vector<node *> next_;
};

}  // namespace litchi