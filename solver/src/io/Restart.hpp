#pragma once

#include "core/Grid.hpp"
#include "core/State.hpp"

#ifdef BLAST_MPI
#include "parallel/Domain.hpp"
#endif

#include <string>

namespace blast {

struct FiveEqAux;   // physics/Multifluid.hpp

// Writes the full interior conserved state plus (time, step) to a single
// HDF5 file. Overwrites the previous checkpoint at the same path -- callers
// typically write into <out_dir>/<run_name>.ckpt.h5 so the latest checkpoint
// is always the same name.
// aux (optional): when non-null, the five-equation aux fields (Z1, Z2, alpha1)
// are written too (and a "five_eq" flag), so a five-equation run is restartable.
void write_checkpoint(const std::string& path, const State& U, const Grid& g,
                      Real t, int step, const FiveEqAux* aux = nullptr);

// Reads (time, step) and the interior conserved state from `path` into U.
// Throws std::runtime_error on grid-shape mismatch.
struct CheckpointHeader {
    Real time;
    int  step;
    int  nx, ny, nz;
};

// aux (optional): if non-null AND the file carries five-equation datasets, the
// aux fields are restored; aux->has_data reports whether they were present.
CheckpointHeader read_checkpoint(const std::string& path, State& U,
                                 const Grid& g, FiveEqAux* aux = nullptr,
                                 bool* aux_restored = nullptr);

#ifdef BLAST_MPI
// MPI variants: collective parallel-HDF5 hyperslab writes and reads. The
// `g` argument is the GLOBAL grid; each rank reads / writes its own
// subblock. Files written at any rank count can be read at any other
// rank count provided the global grid shape matches.
void write_checkpoint(const std::string& path, const State& U,
                      const Grid& global_g, Real t, int step,
                      const Domain& d, const FiveEqAux* aux = nullptr);

CheckpointHeader read_checkpoint(const std::string& path, State& U,
                                 const Grid& global_g, const Domain& d,
                                 FiveEqAux* aux = nullptr,
                                 bool* aux_restored = nullptr);
#endif

}  // namespace blast
