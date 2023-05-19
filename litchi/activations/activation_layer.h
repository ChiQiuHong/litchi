#pragma once

#include "litchi/layers/layer.h"

namespace litchi {

class activation_layer : public layer {
public:
  /**
   * Construct an activation layer with specified width, height and channels.
   * This constructor is suitable for adding an activation layer after spatial
   * layers such as convolution / pooling layers.
   *
   * @param in_width    [in] number of input elements along width
   * @param in_height   [in] number of input elements along height
   * @param in_channels [in] number of channels (input elements along depth)
   */
  activation_layer(size_t in_width, size_t in_height, size_t in_channels)
      : layer(in_channels, in_channels), in_width_(in_width),
        in_height_(in_height), in_channels_(in_channels) {}

  size_t width_shape() const { return in_width_; }
  size_t height_shape() const { return in_height_; }
  size_t channels_shape() const { return in_channels_; }

private:
  size_t in_width_;
  size_t in_height_;
  size_t in_channels_;
};

} // namespace litchi