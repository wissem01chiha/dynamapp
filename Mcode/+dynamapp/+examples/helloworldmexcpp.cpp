/* Hello world in C++ MEX */

#include "mex.hpp"
#include "mexAdapter.hpp"

using namespace matlab::data;
using matlab::mex::ArgumentList;

class MexFunction : public matlab::mex::Function {
  // Sooooo... this MEX function _would_ just output "Hello, world!" on stdout,
  // but it turns out that's non-trivial to do using Matlab's C++ MEX API.
  //   * https://www.mathworks.com/matlabcentral/answers/438100-how-to-use-mexprintf-with-c-mex-functions
  //   * https://www.mathworks.com/help/matlab/matlab_external/displaying-output-in-matlab-command-window.html
  // Go figure.
  // 
  // So, just do nothing.
};
