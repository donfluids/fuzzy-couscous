#include "io/HDF5Writer.hpp"

#include "io/Log.hpp"

#include <hdf5/serial/hdf5.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

namespace blast {

namespace {

void write_dataset_3d(hid_t file, const char* name, int nx, int ny, int nz,
                      const std::vector<double>& data) {
    hsize_t dims[3] = { static_cast<hsize_t>(nz), static_cast<hsize_t>(ny),
                        static_cast<hsize_t>(nx) };
    hid_t space = H5Screate_simple(3, dims, nullptr);
    hid_t dset  = H5Dcreate2(file, name, H5T_NATIVE_DOUBLE, space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Dclose(dset);
    H5Sclose(space);
}

void write_scalar(hid_t file, const char* name, double value) {
    hsize_t dims = 1;
    hid_t space = H5Screate_simple(1, &dims, nullptr);
    hid_t dset  = H5Dcreate2(file, name, H5T_NATIVE_DOUBLE, space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
    H5Dclose(dset);
    H5Sclose(space);
}

void write_dataset_1d(hid_t loc, const char* name,
                      const std::vector<double>& data) {
    hsize_t dims = data.size();
    hid_t space = H5Screate_simple(1, &dims, nullptr);
    hid_t dset  = H5Dcreate2(loc, name, H5T_NATIVE_DOUBLE, space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Dclose(dset);
    H5Sclose(space);
}

}  // namespace

HDF5Writer::HDF5Writer(const std::string& out_dir, const std::string& run_name)
    : out_dir_(out_dir), run_name_(run_name) {
    std::filesystem::create_directories(out_dir_);
}

std::string HDF5Writer::snapshot_path_(int step) const {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s_%06d.h5", run_name_.c_str(), step);
    return out_dir_ + "/" + buf;
}

void HDF5Writer::write_snapshot(const State& U, const Grid& g,
                                const IdealGas& eos, Real t, int step) {
    const int nx = g.nx, ny = g.ny, nz = g.nz;
    const std::size_t N = static_cast<std::size_t>(nx) * ny * nz;

    std::vector<double> rho(N), u(N), v(N), w(N), p(N), T(N);
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const std::size_t idx = static_cast<std::size_t>(i)
                    + nx * (static_cast<std::size_t>(j) + ny * k);
                const Real r  = U[RHO ](i, j, k);
                const Real ui = U[RHOU](i, j, k) / r;
                const Real vi = U[RHOV](i, j, k) / r;
                const Real wi = U[RHOW](i, j, k) / r;
                const Real ke = 0.5 * r * (ui*ui + vi*vi + wi*wi);
                const Real pi = eos.pressure(r, U[RHOE](i, j, k) - ke);
                rho[idx] = r;
                u[idx]   = ui;
                v[idx]   = vi;
                w[idx]   = wi;
                p[idx]   = pi;
                T[idx]   = eos.temperature(r, pi);
            }

    const std::string path = snapshot_path_(step);
    hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) {
        BLAST_ERROR("HDF5: failed to create {}", path);
        return;
    }
    write_scalar(file, "time", t);
    write_scalar(file, "dx",   g.dx());
    write_scalar(file, "dy",   g.dy());
    write_scalar(file, "dz",   g.dz());
    write_scalar(file, "x0",   g.x0);
    write_scalar(file, "y0",   g.y0);
    write_scalar(file, "z0",   g.z0);

    write_dataset_3d(file, "density",     nx, ny, nz, rho);
    write_dataset_3d(file, "velocity_x",  nx, ny, nz, u);
    write_dataset_3d(file, "velocity_y",  nx, ny, nz, v);
    write_dataset_3d(file, "velocity_z",  nx, ny, nz, w);
    write_dataset_3d(file, "pressure",    nx, ny, nz, p);
    write_dataset_3d(file, "temperature", nx, ny, nz, T);
    H5Fclose(file);

    entries_.emplace_back(step, t);
    grid_for_xdmf_ = g;
    update_xdmf_index_();
}

void HDF5Writer::append_spectra(const HelmholtzResult& h,
                                const ShellSpectrum& total,
                                Real t, int step) {
    const std::string path = out_dir_ + "/" + run_name_ + "_spectra.h5";
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) {
        file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    }
    if (file < 0) {
        BLAST_ERROR("HDF5: failed to open or create {}", path);
        return;
    }
    char group_name[64];
    std::snprintf(group_name, sizeof(group_name), "/step_%06d", step);
    hid_t grp = H5Gcreate2(file, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    write_scalar(grp, "time", t);

    auto to_vec = [](const std::vector<Real>& v) {
        return std::vector<double>(v.begin(), v.end());
    };
    write_dataset_1d(grp, "k",       to_vec(total.k));
    write_dataset_1d(grp, "E_total", to_vec(total.E));
    write_dataset_1d(grp, "E_sol",   to_vec(h.E_sol.E));
    write_dataset_1d(grp, "E_dil",   to_vec(h.E_dil.E));

    H5Gclose(grp);
    H5Fclose(file);
}

void HDF5Writer::update_xdmf_index_() {
    // ParaView/VisIt-compatible XDMF temporal collection over a uniform
    // rectilinear cell-centered grid. HDF5 datasets are stored with shape
    // (nz, ny, nx); for cell-centered fields the topology dimensions are
    // (nz+1, ny+1, nx+1).
    const std::string path = out_dir_ + "/" + run_name_ + ".xdmf";
    std::ofstream f(path);
    if (!f) {
        BLAST_ERROR("HDF5: failed to open XDMF index {}", path);
        return;
    }
    const Grid& g = grid_for_xdmf_;
    const int nx = g.nx, ny = g.ny, nz = g.nz;

    auto write_attribute = [&](const char* name, const std::string& snap) {
        f << "        <Attribute Name=\"" << name
          << "\" AttributeType=\"Scalar\" Center=\"Cell\">\n";
        f << "          <DataItem Dimensions=\"" << nz << " " << ny << " " << nx
          << "\" Format=\"HDF\" NumberType=\"Float\" Precision=\"8\">\n";
        f << "            " << snap << ":/" << name << "\n";
        f << "          </DataItem>\n";
        f << "        </Attribute>\n";
    };

    f << "<?xml version=\"1.0\" ?>\n";
    f << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
    f << "<Xdmf Version=\"2.2\">\n";
    f << "  <Domain>\n";
    f << "    <Grid Name=\"snapshots\" GridType=\"Collection\" "
         "CollectionType=\"Temporal\">\n";
    for (auto [step, t] : entries_) {
        char snap[64];
        std::snprintf(snap, sizeof(snap), "%s_%06d.h5", run_name_.c_str(), step);
        f << "      <Grid Name=\"step_" << step << "\" GridType=\"Uniform\">\n";
        f << "        <Time Value=\"" << t << "\"/>\n";
        f << "        <Topology TopologyType=\"3DCoRectMesh\" Dimensions=\""
          << (nz + 1) << " " << (ny + 1) << " " << (nx + 1) << "\"/>\n";
        f << "        <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n";
        f << "          <DataItem Dimensions=\"3\" Format=\"XML\">"
          << g.z0 << " " << g.y0 << " " << g.x0 << "</DataItem>\n";
        f << "          <DataItem Dimensions=\"3\" Format=\"XML\">"
          << g.dz() << " " << g.dy() << " " << g.dx() << "</DataItem>\n";
        f << "        </Geometry>\n";
        for (const char* name : {"density", "velocity_x", "velocity_y",
                                  "velocity_z", "pressure", "temperature"}) {
            write_attribute(name, snap);
        }
        f << "      </Grid>\n";
    }
    f << "    </Grid>\n";
    f << "  </Domain>\n";
    f << "</Xdmf>\n";
}

}  // namespace blast
