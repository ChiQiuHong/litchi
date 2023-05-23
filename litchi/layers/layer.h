#pragma once

#include <memory>

#include "litchi/core/backend.h"
#include "litchi/node.h"

#include "litchi/util/util.h"
#include "litchi/util/weight_init.h"

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
    : node(in_type.size(), out_type.size()),
      initialized_(false),
      in_channels_(in_type.size()),
      out_channels_(out_type.size()),
      in_type_(in_type),
      out_type_(out_type) {
    weight_init_ = std::make_shared<weight_init::xavier>();
    bias_init_   = std::make_shared<weight_init::constant>();
    trainable_   = true;
  }

  // void set_backend(std::shared_ptr<core::backend> backend) {
  //   backend_ = backend;
  // }

  void set_backend_type(core::backend_t backend_type) {
    backend_type_ = backend_type;
  }

  ///////////////////////////////////////////////////////////
  // getter

  // TODO (edgar): Deprecated: use the below method

  core::backend_t engine() const { return backend_type_; }

  ///< number of incoming edges in this layer
  size_t in_channels() const { return in_channels_; }

  void set_in_data(const std::vector<const vec_t *> *data, size_t cnt) {
    CNN_UNREFERENCED_PARAMETER(cnt);
    size_t n = 0;
    for (size_t i = 0; i < in_channels_; i++) {
      if (in_type_[i] != vector_type::data) continue;
      tensor_t &dst_data = *ith_in_node(i)->get_data();
      size_t in_size     = ith_in_node(i)->shape().size();
      assert(n < cnt);
      const auto &src_data = data[n++];
      size_t sz            = src_data.size();
      dst_data.resize(sz);

      CNN_UNREFERENCED_PARAMETER(in_size);

      for (size_t j = 0; j < sz; ++j) {
        assert(
          src_data[j]->size() ==
          in_size);  // checking if training data is consistent with layer shape
        dst_data[j] = *src_data[j];
      }
    }
  }

  void output(std::vector<const tensor_t *> &out) const {
    out.clear();
    for (size_t i = 0; i < out_channels_; i++) {
      if (out_type_[i] == vector_type::data) {
        out.push_back(ith_out_node(i)->get_data());
      }
    }
  }

  /**
   * array of input shapes (width x height x depth)
   */
  virtual std::vector<shape3d> in_shape() const = 0;

  /**
   * array of output shapes (width x height x depth)
   */
  virtual std::vector<shape3d> out_shape() const = 0;

  /**
   * number of incoming connections for each output unit
   * used only for weight/bias initialization methods which require fan-in
   * size
   * (e.g. xavier)
   * override if the layer has trainable weights, and scale of initialization
   * is
   * important
   */
  virtual size_t fan_in_size() const { return in_shape()[0].width_; }
  // override to allow initialization of multiple size weight matrices.
  virtual size_t fan_in_size(size_t) const {
    return fan_in_size();  // fallback to single weight matrix.
  }

  /**
   * number of outgoing connections for each output unit
   * used only for weight/bias initialization methods which require fan-in
   * size
   * (e.g. xavier)
   * override if the layer has trainable weights, and scale of initialization
   * is
   * important
   */
  virtual size_t fan_out_size() const { return out_shape()[0].width_; }
  // override to allow initialization of multiple size weight matrices.
  virtual size_t fan_out_size(size_t) const {
    return fan_out_size();  // fallback to single weight matrix.
  }

  //////////////////////////////////////////////////////////////////////////
  // setter
  template <typename WeightInit>
  layer &weight_init(const WeightInit &f) {
    weight_init_ = std::make_shared<WeightInit>(f);
    return *this;
  }

  template <typename BiasInit>
  layer &bias_init(const BiasInit &f) {
    bias_init_ = std::make_shared<BiasInit>(f);
    return *this;
  }

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

  /**
   * @brief Performs layer forward operation given an input tensor and
   * returns the computed data in tensor form.
   *
   * @param input vector of `tensor_t` with incoming data.
   *
   * Internally, it first allocates data without resetting the weights,
   * forwards the input data to the computational graph, inside the
   * forward() method the data from the computational embedded to container
   * to finally be forwarded to the computational operation kernels.
   *
   * // TODO: Probably there's an overhead of moving from/to the computational
   * graph. Will be this overhead reduced once we have the Tensor class
   * integrated?
   */
  void forward(const std::vector<tensor_t> &input,
               std::vector<const tensor_t *> &out) {  // for test
    // allocate data in the computational graph without
    // resetting the weights.
    setup(false);

    std::vector<std::vector<const vec_t *>> input2;
    input2.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
      input2[i].resize(input[i].size());
      for (size_t j = 0; j < input[i].size(); ++j) {
        input2[i][j] = &input[i][j];
      }
    }

    // the incoming data is forwarded to the computational graph.
    set_in_data(&input2[0], input2.size());
    // pick up the data from the computational graph and perform
    // computation.
    forward();
    // retrieve computed data and return values in form of 4D tensor.
    output(out);
  }

  void forward() {
    // the  computational graph
    fwd_in_data_.resize(in_channels_);
    fwd_out_data_.resize(out_channels_);

    // Organize input/output vectors from storage (computational graph).
    // Internally ith_in_node() will create a connection/edge in the
    // computational graph and will allocate memory in case that it's not
    // done yet.
    for (size_t i = 0; i < in_channels_; i++) {
      fwd_in_data_[i] = ith_in_node(i)->get_data();
    }

    // resize outs and stuff to have room for every input sample in
    // the batch
    set_sample_count(fwd_in_data_[0]->size());

    // Internally ith_out_node() will create a connection/edge to the
    // computational graph and will allocate memory in case that it's not
    // done yet. In addition, gradient vector are initialized to default
    // values.
    for (size_t i = 0; i < out_channels_; i++) {
      fwd_out_data_[i] = ith_out_node(i)->get_data();
      ith_out_node(i)->clear_grads();
    }

    // call the forward computation kernel/routine
    forward_propagation(fwd_in_data_, fwd_out_data_);
  }

  /**
   * @brief Allocates data in the computational graph and reset weights if
   * it's needed or the data is not already initialized.
   *
   * @param reset_weight Boolean value to force to reset the weights.
   * Weights will be automatically reset if the are not initialized.
   */
  void setup(bool reset_weight) {
    // The input shape (width x height x depth) must be equal to the number
    // of input channels a.k.a the number of incoming vectors or 'edges' in
    // the computational nomenclature. Same is applied to output shape and
    // numbers of output edges.
    if (in_shape().size() != in_channels_ ||
        out_shape().size() != out_channels_) {
      throw "Connection mismatch at setup layer";
    }

    // An 'edge' is created in the computational graph from the current
    // layer/node to each output node and allocates the needed memory.
    // The number of output nodes is determined by the layer interface.
    // In order to handle graph based networks, which a layer/node might
    // have multiple input/output connections, we need to check that the
    // connection edge does not already exists if we don't want duplicated
    // memory allocation.
    for (size_t i = 0; i < out_channels_; i++) {
      if (!next_[i]) {
        // connection edge doesn't exists, so we proceed to allocate the
        // necessary memory.
        next_[i] = std::make_shared<edge>(this, out_shape()[i], out_type_[i]);
      }
    }

    // reset the weights if necessary, or in case that the data is
    // still not initialized.
    if (reset_weight || !initialized_) {
      init_weight();
    }
  }

  /**
   * @brief Initializes the vectors containing the trainable data.
   *
   * In case that a layer/node is set to be not trainable, it does
   * nothing and returns a void. Otherwise, for each input connection
   * and depending of the data nature (weight or bias) calls their
   * pertinent initialization function and fill the vectors with the
   * data generated by the mentioned functions.
   */
  void init_weight() {
    // layer/node is not trainable, do nothing and mark the layer/node
    // as initialized.
    if (!trainable_) {
      initialized_ = true;
      return;
    }

    // Fill vector values with data generated by the initialization
    // function. The pointer to the data is obtained from the
    // computational graph and the methods fan_in_size() and fan_out_size()
    // return the number of incoming/outcoming connections for each
    // ipnut/output unit.
    for (size_t i = 0; i < in_channels_; i++) {
      switch (in_type_[i]) {
        // fill vectors of weight type
        case vector_type::weight:
          weight_init_->fill(get_weight_data(i), fan_in_size(i),
                             fan_out_size(i));
          break;
        // fill vector of bias type
        case vector_type::bias:
          bias_init_->fill(get_weight_data(i), fan_in_size(i), fan_out_size(i));
          break;
        default: break;
      }
    }
    // in case we succeed with data initialization, we mark the
    // layer/node as initialized.
    initialized_ = true;
  }

  virtual void set_sample_count(size_t sample_count) {
    // increase the size if necessary - but do not decrease
    auto resize = [sample_count](tensor_t *tensor) {
      tensor->resize(sample_count, (*tensor)[0]);
    };

    for (size_t i = 0; i < in_channels_; i++) {
      if (!is_trainable_weight(in_type_[i])) {
        resize(ith_in_node(i)->get_data());
      }
      resize(ith_in_node(i)->get_gradient());
    }

    for (size_t i = 0; i < out_channels_; i++) {
      if (!is_trainable_weight(out_type_[i])) {
        resize(ith_out_node(i)->get_data());
      }
      resize(ith_out_node(i)->get_gradient());
    }
  }

 protected:
  /** Flag indication whether the layer/node is initialized */
  bool initialized_;
  /** The number of input vectors/edges */
  size_t in_channels_;
  /** The number of output vectors/edges */
  size_t out_channels_;
  /** Vector containing the type of data for inputs */
  std::vector<vector_type> in_type_;
  /** Vector containing the type of data for outputs */
  std::vector<vector_type> out_type_;
  /** The current backend type for operations */
  core::backend_t backend_type_;
  /** The backend instance (deprecated) */
  // std::shared_ptr<core::backend> backend_;

 private:
  /** Flag indicating whether the layer/node parameters are trainable */
  bool trainable_;
  /** Pointer to the function for weights initialization */
  std::shared_ptr<weight_init::function> weight_init_;
  /** Pointer to the function for biases initialization */
  std::shared_ptr<weight_init::function> bias_init_;

  std::vector<tensor_t *> fwd_in_data_;
  std::vector<tensor_t *> fwd_out_data_;

  /**
   * @brief Allocates the necessary edge memory in a specific
   * incoming connection.
   *
   * @param i The position to store the previous edge.
   *
   * Graphical explanation:
   *
   *     nullptr -- |edge| -- prev(i) ---- |layer|
   *               nullptr -- prev(i+1) - `
   */
  void alloc_input(size_t i) const {
    // the created incoming edge won't have a previous connection,
    // for this reason first parameter is a nullptr.
    prev_[i] = std::make_shared<edge>(nullptr, in_shape()[i], in_type_[i]);
  }

  /**
   * @brief Allocates the necessary edge memory in a specific
   * outcoming connection.
   *
   * @param i The position to store the next edge.
   *
   * Graphical explanation:
   *
   *     |layer| -- next(i) ---- |edge|
   *             `- next(i+1) -- nullptr
   */
  void alloc_output(size_t i) const {
    // the created outcoming will have the current layer as the
    // previous node.
    next_[i] = std::make_shared<edge>(const_cast<layer *>(this), out_shape()[i],
                                      out_type_[i]);
  }

  /**
   * @brief Creates an edge between the current node and one incoming
   * or previous node.
   *
   * @param i The position to store the previous edge.
   *
   * The method checks if the edge already exists, otherwise we create it
   * and the necessary memory it's allocated. The method returns the pointer
   * to the previous edge.
   */
  edgeptr_t ith_in_node(size_t i) {
    // in case that the edge doesn't exist, we create it
    if (!prev_[i]) alloc_input(i);
    return prev()[i];
  }

  /**
   * @brief Creates an edge between the current node and one outcoming
   * or next node.
   *
   * @param i The position to store the next edge.
   *
   * The method checks if the edge already exists, otherwise we create it
   * and the necessary memory it's allocated. The method returns the pointer
   * to the next edge.
   */
  edgeptr_t ith_out_node(size_t i) {
    // in case that the edge doesn't exist, we create it
    if (!next_[i]) alloc_output(i);
    return next()[i];
  }
  edgeptr_t ith_out_node(size_t i) const { return next()[i]; }

  /**
   * @brief Retrieves weight vector from incoming edge
   * @param i The position of incoming edge.
   *
   * Returns the mutable pointer to the edge raw data.
   */
  vec_t *get_weight_data(size_t i) {
    assert(is_trainable_weight(in_type_[i]));
    return &(*(ith_in_node(i)->get_data()))[0];
  }

  /**
   * @brief Retrieves weight vector from incoming edge
   * @param i The position of incoming edge.
   *
   * Returns the mutable pointer to the edge raw data.
   */
  const vec_t *get_weight_data(size_t i) const {
    assert(is_trainable_weight(in_type_[i]));
    return &(*(const_cast<layer *>(this)->ith_in_node(i)->get_data()))[0];
  }
};

}  // namespace litchi