#pragma once

#include "core/Grid.hpp"
#include "core/State.hpp"

#include <string>

namespace blast {

// Writes the full interior conserved state plus (time, step) to a single
// HDF5 file. Overwrites the previous checkpoint at the same path -- callers
// typically write into <out_dir>/<run_name>.ckpt.h5 so the latest checkpoint
// is always the same name.
void write_checkpoint(const std::string& path, const State& U, const Grid& g,
                      Real t, int step);

// Reads (time, step) and the interior conserved state from `path` into U.
// Throws std::runtime_error on grid-shape mismatch.
struct CheckpointHeader {
    Real time;
    int  step;
    int  nx, ny, nz;
};

CheckpointHeader read_checkpoint(const std::string& path, State& U,
                                 const Grid& g);

}  // namespace blast
