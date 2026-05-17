#include "io/Restart.hpp"

#include "io/Log.hpp"

#include <hdf5.h>

#include <stdexcept>
#include <vector>

namespace blast {

namespace {

void write_scalar_d(hid_t file, const char* name, double v) {
    hsize_t one = 1;
    hid_t s = H5Screate_simple(1, &one, nullptr);
    hid_t d = H5Dcreate2(file, name, H5T_NATIVE_DOUBLE, s,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v);
    H5Dclose(d); H5Sclose(s);
}

void write_scalar_i(hid_t file, const char* name, int v) {
    hsize_t one = 1;
    hid_t s = H5Screate_simple(1, &one, nullptr);
    hid_t d = H5Dcreate2(file, name, H5T_NATIVE_INT, s,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(d, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v);
    H5Dclose(d); H5Sclose(s);
}

double read_scalar_d(hid_t file, const char* name) {
    hid_t d = H5Dopen2(file, name, H5P_DEFAULT);
    double v;
    H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v);
    H5Dclose(d);
    return v;
}

int read_scalar_i(hid_t file, const char* name) {
    hid_t d = H5Dopen2(file, name, H5P_DEFAULT);
    int v;
    H5Dread(d, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v);
    H5Dclose(d);
    return v;
}

void pack_interior(const Field3D& f, std::vector<double>& out) {
    const int nx = f.nx(), ny = f.ny(), nz = f.nz();
    out.resize(static_cast<std::size_t>(nx) * ny * nz);
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const std::size_t idx = static_cast<std::size_t>(i)
                    + nx * (static_cast<std::size_t>(j) + ny * k);
                out[idx] = f(i, j, k);
            }
}

void unpack_interior(const std::vector<double>& in, Field3D& f) {
    const int nx = f.nx(), ny = f.ny(), nz = f.nz();
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const std::size_t idx = static_cast<std::size_t>(i)
                    + nx * (static_cast<std::size_t>(j) + ny * k);
                f(i, j, k) = in[idx];
            }
}

}  // namespace

void write_checkpoint(const std::string& path, const State& U, const Grid& g,
                      Real t, int step) {
    hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) {
        BLAST_ERROR("checkpoint: failed to open {} for write", path);
        return;
    }
    write_scalar_d(file, "time", t);
    write_scalar_i(file, "step", step);
    write_scalar_i(file, "nx",   g.nx);
    write_scalar_i(file, "ny",   g.ny);
    write_scalar_i(file, "nz",   g.nz);

    static const char* names[NCONS] = {"rho", "rho_u", "rho_v", "rho_w", "rho_E"};
    std::vector<double> buf;
    for (int v = 0; v < NCONS; ++v) {
        pack_interior(U[v], buf);
        hsize_t dims[3] = { static_cast<hsize_t>(g.nz),
                            static_cast<hsize_t>(g.ny),
                            static_cast<hsize_t>(g.nx) };
        hid_t space = H5Screate_simple(3, dims, nullptr);
        hid_t dset  = H5Dcreate2(file, names[v], H5T_NATIVE_DOUBLE, space,
                                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 buf.data());
        H5Dclose(dset);
        H5Sclose(space);
    }
    H5Fclose(file);
}

CheckpointHeader read_checkpoint(const std::string& path, State& U,
                                 const Grid& g) {
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        throw std::runtime_error("checkpoint: failed to open " + path);
    }
    CheckpointHeader h;
    h.time = read_scalar_d(file, "time");
    h.step = read_scalar_i(file, "step");
    h.nx   = read_scalar_i(file, "nx");
    h.ny   = read_scalar_i(file, "ny");
    h.nz   = read_scalar_i(file, "nz");

    if (h.nx != g.nx || h.ny != g.ny || h.nz != g.nz) {
        H5Fclose(file);
        throw std::runtime_error(
            "checkpoint grid shape (" + std::to_string(h.nx) + "x"
            + std::to_string(h.ny) + "x" + std::to_string(h.nz)
            + ") does not match config (" + std::to_string(g.nx) + "x"
            + std::to_string(g.ny) + "x" + std::to_string(g.nz) + ")");
    }

    static const char* names[NCONS] = {"rho", "rho_u", "rho_v", "rho_w", "rho_E"};
    std::vector<double> buf(static_cast<std::size_t>(g.nx) * g.ny * g.nz);
    for (int v = 0; v < NCONS; ++v) {
        hid_t dset = H5Dopen2(file, names[v], H5P_DEFAULT);
        H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                buf.data());
        H5Dclose(dset);
        unpack_interior(buf, U[v]);
    }
    H5Fclose(file);
    return h;
}

#ifdef BLAST_MPI

namespace {

// Collective-safe scalar write: every rank calls it; only rank 0 writes data.
void write_scalar_d_collective(hid_t file, const char* name, double v,
                                int rank) {
    hsize_t one = 1;
    hid_t s = H5Screate_simple(1, &one, nullptr);
    hid_t dset = H5Dcreate2(file, name, H5T_NATIVE_DOUBLE, s,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_INDEPENDENT);
    if (rank == 0) H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, dxpl, &v);
    H5Pclose(dxpl);
    H5Dclose(dset);
    H5Sclose(s);
}

void write_scalar_i_collective(hid_t file, const char* name, int v,
                                int rank) {
    hsize_t one = 1;
    hid_t s = H5Screate_simple(1, &one, nullptr);
    hid_t dset = H5Dcreate2(file, name, H5T_NATIVE_INT, s,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_INDEPENDENT);
    if (rank == 0) H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, dxpl, &v);
    H5Pclose(dxpl);
    H5Dclose(dset);
    H5Sclose(s);
}

}  // namespace

void write_checkpoint(const std::string& path, const State& U,
                      const Grid& global_g, Real t, int step,
                      const Domain& d) {
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl, d.comm(), MPI_INFO_NULL);
    hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    if (file < 0) {
        if (d.rank() == 0)
            BLAST_ERROR("checkpoint(MPI): failed to open {} for write", path);
        return;
    }

    write_scalar_d_collective(file, "time", t,           d.rank());
    write_scalar_i_collective(file, "step", step,        d.rank());
    write_scalar_i_collective(file, "nx",   global_g.nx, d.rank());
    write_scalar_i_collective(file, "ny",   global_g.ny, d.rank());
    write_scalar_i_collective(file, "nz",   global_g.nz, d.rank());

    const auto off = d.global_offset(global_g);
    hsize_t file_dims[3] = {
        static_cast<hsize_t>(global_g.nz),
        static_cast<hsize_t>(global_g.ny),
        static_cast<hsize_t>(global_g.nx)
    };
    hsize_t mem_dims[3] = {
        static_cast<hsize_t>(U.nz()),
        static_cast<hsize_t>(U.ny()),
        static_cast<hsize_t>(U.nx())
    };
    hsize_t file_offset[3] = {
        static_cast<hsize_t>(off[2]),
        static_cast<hsize_t>(off[1]),
        static_cast<hsize_t>(off[0])
    };

    hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);

    static const char* names[NCONS] = {"rho", "rho_u", "rho_v", "rho_w", "rho_E"};
    std::vector<double> buf;
    for (int v = 0; v < NCONS; ++v) {
        pack_interior(U[v], buf);
        hid_t fspace = H5Screate_simple(3, file_dims, nullptr);
        hid_t dset = H5Dcreate2(file, names[v], H5T_NATIVE_DOUBLE, fspace,
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Sselect_hyperslab(fspace, H5S_SELECT_SET, file_offset, nullptr,
                            mem_dims, nullptr);
        hid_t mspace = H5Screate_simple(3, mem_dims, nullptr);
        H5Dwrite(dset, H5T_NATIVE_DOUBLE, mspace, fspace, dxpl, buf.data());
        H5Sclose(mspace);
        H5Dclose(dset);
        H5Sclose(fspace);
    }
    H5Pclose(dxpl);
    H5Fclose(file);
}

CheckpointHeader read_checkpoint(const std::string& path, State& U,
                                 const Grid& global_g, const Domain& d) {
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl, d.comm(), MPI_INFO_NULL);
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    if (file < 0) {
        throw std::runtime_error("checkpoint(MPI): failed to open " + path);
    }

    CheckpointHeader h;
    h.time = read_scalar_d(file, "time");
    h.step = read_scalar_i(file, "step");
    h.nx   = read_scalar_i(file, "nx");
    h.ny   = read_scalar_i(file, "ny");
    h.nz   = read_scalar_i(file, "nz");

    if (h.nx != global_g.nx || h.ny != global_g.ny || h.nz != global_g.nz) {
        H5Fclose(file);
        throw std::runtime_error(
            "checkpoint(MPI) grid shape (" + std::to_string(h.nx) + "x"
            + std::to_string(h.ny) + "x" + std::to_string(h.nz)
            + ") does not match config (" + std::to_string(global_g.nx) + "x"
            + std::to_string(global_g.ny) + "x" + std::to_string(global_g.nz) + ")");
    }

    const auto off = d.global_offset(global_g);
    hsize_t mem_dims[3] = {
        static_cast<hsize_t>(U.nz()),
        static_cast<hsize_t>(U.ny()),
        static_cast<hsize_t>(U.nx())
    };
    hsize_t file_offset[3] = {
        static_cast<hsize_t>(off[2]),
        static_cast<hsize_t>(off[1]),
        static_cast<hsize_t>(off[0])
    };

    hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);

    static const char* names[NCONS] = {"rho", "rho_u", "rho_v", "rho_w", "rho_E"};
    std::vector<double> buf(static_cast<std::size_t>(U.nx()) * U.ny() * U.nz());
    for (int v = 0; v < NCONS; ++v) {
        hid_t dset = H5Dopen2(file, names[v], H5P_DEFAULT);
        hid_t fspace = H5Dget_space(dset);
        H5Sselect_hyperslab(fspace, H5S_SELECT_SET, file_offset, nullptr,
                            mem_dims, nullptr);
        hid_t mspace = H5Screate_simple(3, mem_dims, nullptr);
        H5Dread(dset, H5T_NATIVE_DOUBLE, mspace, fspace, dxpl, buf.data());
        H5Sclose(mspace);
        H5Sclose(fspace);
        H5Dclose(dset);
        unpack_interior(buf, U[v]);
    }
    H5Pclose(dxpl);
    H5Fclose(file);
    return h;
}

#endif

}  // namespace blast
