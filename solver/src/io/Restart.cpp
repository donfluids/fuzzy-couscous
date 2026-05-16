#include "io/Restart.hpp"

#include "io/Log.hpp"

#include <hdf5/serial/hdf5.h>

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

}  // namespace blast
