#pragma once

#include <vector>

namespace litchi {

TEST(relu, gradient_check) {
  const size_t width = 3;
  const size_t height = 3;
  const size_t channels = 10;
  relu rlu(width, height, channels);

  std::vector<float> x{-0.1, -0.2, 0.3, 0.4, 0.2, -0.5};
  std::vector<float> y(6);

  std::vector<float> z{0, 0, 0.3, 0.4, 0.2, 0};

  rlu.forward_activation(x, y);

  EXPECT_EQ(y, z);
}

} // namespace litchi