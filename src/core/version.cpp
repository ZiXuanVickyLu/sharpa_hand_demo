#include "version.h"

namespace cs {

std::string version_string() {
#ifdef CS_USE_DOUBLE
    return "0.1.0-dev (real=double)";
#else
    return "0.1.0-dev (real=float)";
#endif
}

bool built_with_double() {
#ifdef CS_USE_DOUBLE
    return true;
#else
    return false;
#endif
}

}  // namespace cs
