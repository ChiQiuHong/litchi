#pragma once

namespace litchi {

namespace core {

class fully_params;

/* Base class to model operation parameters */
class Params {
public:
  Params() {}

  fully_params &fully();
};

}  // namespace core

}  // namespace litchi