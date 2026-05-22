#pragma once

#include "core/Grid.hpp"
#include "core/State.hpp"
#include "diagnostics/Spectra.hpp"
#include "physics/EOS.hpp"
#include "physics/MixtureEOS.hpp"

#ifdef BLAST_MPI
#include "parallel/Domain.hpp"
#include <mpi.h>
#endif

#include <string>
#include <utility>
#include <vector>

namespace blast {

// Writes a single snapshot (rho, u, v, w, p, T) to an HDF5 file. The file
// name is `<dir>/snapshot_<step:06d>.h5`. Each call independently opens and
// closes the file; the companion XDMF index is regenerated on every call so
// ParaView/VisIt picks up new snapshots without restarting.
//
// MPI build: pass a Domain via set_domain() before writing. Snapshots are
// written collectively via parallel HDF5 hyperslabs; the dataset shape is
// the GLOBAL grid (nz_g, ny_g, nx_g) and each rank writes its own subblock.
class HDF5Writer {
public:
    HDF5Writer(const std::string& out_dir, const std::string& run_name);

    // gfn (optional): multifluid marker field. When provided, the snapshot
    // pressure uses the local EOS: two-gamma p=(rhoE-ke)/G, or (with a JWL mix)
    // the JWL products EOS. Temperature uses the ideal-gas relation (a JWL
    // products temperature needs a caloric model -- placeholder for now).
    void write_snapshot(const State& U, const Grid& g, const IdealGas& eos,
                        Real t, int step, const Field3D* gfn = nullptr,
                        const MixtureEOS* mix = nullptr);

    // Append per-step spectra to <run_name>_spectra.h5 under /step_<NNNNNN>/.
    // Each entry holds time, k, E_total, E_sol, E_dil 1D datasets.
    void append_spectra(const HelmholtzResult& h, const ShellSpectrum& total,
                        Real t, int step);

#ifdef BLAST_MPI
    // Bind to a Domain + Cart comm; subsequent write_snapshot calls switch
    // to collective parallel-HDF5 hyperslab writes. The `g` passed to
    // write_snapshot must be the GLOBAL grid in MPI mode.
    void set_domain(const Domain* d) { domain_ = d; }
#endif

private:
    std::string out_dir_;
    std::string run_name_;
    std::vector<std::pair<int, Real>> entries_;   // step, time
    Grid grid_for_xdmf_{};                        // captured on first snapshot

#ifdef BLAST_MPI
    const Domain* domain_ = nullptr;
#endif

    void update_xdmf_index_();
    std::string snapshot_path_(int step) const;
};

}  // namespace blast
