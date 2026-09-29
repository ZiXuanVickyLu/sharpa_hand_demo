#pragma once
#include <string>

namespace cs {

// Semantic version of the solver library plus the precision it was built with.
std::string version_string();

// True when `real` is double (CS_USE_DOUBLE).
bool built_with_double();

}  // namespace cs
