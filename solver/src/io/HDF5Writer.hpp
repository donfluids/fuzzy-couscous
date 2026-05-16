#pragma once

#include "core/Grid.hpp"
#include "core/State.hpp"
#include "physics/EOS.hpp"

#include <string>
#include <utility>
#include <vector>

namespace blast {

// Writes a single snapshot (rho, u, v, w, p, T) to an HDF5 file. The file
// name is `<dir>/snapshot_<step:06d>.h5`. Each call independently opens and
// closes the file; the companion XDMF index is regenerated on every call so
// ParaView/VisIt picks up new snapshots without restarting.
class HDF5Writer {
public:
    HDF5Writer(const std::string& out_dir, const std::string& run_name);

    void write_snapshot(const State& U, const Grid& g, const IdealGas& eos,
                        Real t, int step);

private:
    std::string out_dir_;
    std::string run_name_;
    std::vector<std::pair<int, Real>> entries_;   // step, time

    void update_xdmf_index_();
    std::string snapshot_path_(int step) const;
};

}  // namespace blast
