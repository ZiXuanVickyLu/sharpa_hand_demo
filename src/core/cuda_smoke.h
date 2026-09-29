#pragma once

namespace cs {

// Launches a trivial kernel that exercises `real` arithmetic, the vector helpers and a
// device-side reduction, and checks the result on the host. Returns 0 on success.
int cuda_smoke();

}  // namespace cs
