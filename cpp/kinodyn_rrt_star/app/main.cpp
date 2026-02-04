// A minimal CLI runner that accepts start/goal/params from command line, and an optional grid map file.
// Map file format (ASCII):
//  Line 1: nx ny nz
//  Line 2: resolution
//  Line 3: origin_x origin_y origin_z
//  Following: nx*ny*nz integers (0 or 1), whitespace separated, row-major [x,y,z]

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <limits>
#include <Eigen/Dense>
#include "kinodyn_rrt_star/kinodyn_rrt_star.h"

struct GridMap : public OccupancyMapInterface {
  int nx{0}, ny{0}, nz{0};
  double resolution{0.1};
  Eigen::Vector3d origin{Eigen::Vector3d::Zero()};
  std::vector<uint8_t> data; // 0 free, 1 occupied

  bool hasObservation() const override {
    return nx > 0 && ny > 0 && nz > 0 && !data.empty();
  }

  inline bool inBounds(int ix, int iy, int iz) const {
    return ix >= 0 && iy >= 0 && iz >= 0 && ix < nx && iy < ny && iz < nz;
  }

  bool isOccupied(const Eigen::Vector3d &p) const override {
    Eigen::Vector3d rel = p - origin; // origin is map center
    double hx = (nx * resolution) * 0.5;
    double hy = (ny * resolution) * 0.5;
    double hz = (nz * resolution) * 0.5;
    if (std::abs(rel.x()) > hx || std::abs(rel.y()) > hy || std::abs(rel.z()) > hz)
      return true; // outside map treated as occupied
    int ix = static_cast<int>(std::floor((rel.x() + hx) / resolution));
    int iy = static_cast<int>(std::floor((rel.y() + hy) / resolution));
    int iz = static_cast<int>(static_cast<int>(std::floor((rel.z() + hz) / resolution)));
    if (!inBounds(ix, iy, iz)) return true;
    size_t idx = static_cast<size_t>(ix) + static_cast<size_t>(nx) * (static_cast<size_t>(iy) + static_cast<size_t>(ny) * static_cast<size_t>(iz));
    return data[idx] != 0;
  }
  double distanceAt(const Eigen::Vector3d &) const override { return std::numeric_limits<double>::infinity(); }
};

// DualGridMap: occupancy + optional ESDF (ASCII), with inflation radius
struct DualGridMap : public OccupancyMapInterface {
  int nx{0}, ny{0}, nz{0};
  double resolution{0.1};
  Eigen::Vector3d origin{Eigen::Vector3d::Zero()};
  std::vector<uint8_t> occ; // 0 free, 1 occupied
  std::vector<float> esdf;  // meters, signed; optional
  bool has_esdf{false};
  double inflate_radius{0.0};

  bool hasObservation() const override {
    return nx > 0 && ny > 0 && nz > 0 && !occ.empty();
  }

  inline bool inBounds(int ix, int iy, int iz) const {
    return ix >= 0 && iy >= 0 && iz >= 0 && ix < nx && iy < ny && iz < nz;
  }

  inline size_t index(int ix, int iy, int iz) const {
    return static_cast<size_t>(ix) + static_cast<size_t>(nx) * (static_cast<size_t>(iy) + static_cast<size_t>(ny) * static_cast<size_t>(iz));
  }

  double esdfAt(const Eigen::Vector3d &p) const {
    if (!has_esdf) return std::numeric_limits<double>::infinity();
    Eigen::Vector3d rel = p - origin;
    double hx = (nx * resolution) * 0.5;
    double hy = (ny * resolution) * 0.5;
    double hz = (nz * resolution) * 0.5;
    if (std::abs(rel.x()) > hx || std::abs(rel.y()) > hy || std::abs(rel.z()) > hz)
      return -1e9; // outside
    double fx = (rel.x() + hx) / resolution;
    double fy = (rel.y() + hy) / resolution;
    double fz = (rel.z() + hz) / resolution;
    int x0 = static_cast<int>(std::floor(fx));
    int y0 = static_cast<int>(std::floor(fy));
    int z0 = static_cast<int>(std::floor(fz));
    int x1 = std::min(x0 + 1, nx - 1);
    int y1 = std::min(y0 + 1, ny - 1);
    int z1 = std::min(z0 + 1, nz - 1);
    double xd = fx - x0, yd = fy - y0, zd = fz - z0;
    auto at = [&](int ix, int iy, int iz){ return static_cast<double>(esdf[index(ix,iy,iz)]); };
    double c000 = at(x0,y0,z0), c100 = at(x1,y0,z0);
    double c010 = at(x0,y1,z0), c110 = at(x1,y1,z0);
    double c001 = at(x0,y0,z1), c101 = at(x1,y0,z1);
    double c011 = at(x0,y1,z1), c111 = at(x1,y1,z1);
    double c00 = c000 * (1 - xd) + c100 * xd;
    double c01 = c001 * (1 - xd) + c101 * xd;
    double c10 = c010 * (1 - xd) + c110 * xd;
    double c11 = c011 * (1 - xd) + c111 * xd;
    double c0 = c00 * (1 - yd) + c10 * yd;
    double c1 = c01 * (1 - yd) + c11 * yd;
    return c0 * (1 - zd) + c1 * zd;
  }

  bool isOccupied(const Eigen::Vector3d &p) const override {
    Eigen::Vector3d rel = p - origin; // origin is map center
    double hx = (nx * resolution) * 0.5;
    double hy = (ny * resolution) * 0.5;
    double hz = (nz * resolution) * 0.5;
    if (std::abs(rel.x()) > hx || std::abs(rel.y()) > hy || std::abs(rel.z()) > hz)
      return true; // outside map treated as occupied

    if (has_esdf && inflate_radius > 0.0) {
      double d = esdfAt(p);
      if (d < inflate_radius) return true;
    }

    int ix = static_cast<int>(std::floor((rel.x() + hx) / resolution));
    int iy = static_cast<int>(std::floor((rel.y() + hy) / resolution));
    int iz = static_cast<int>(std::floor((rel.z() + hz) / resolution));
    if (!inBounds(ix, iy, iz)) return true;
    return occ[index(ix,iy,iz)] != 0;
  }
  double distanceAt(const Eigen::Vector3d &p) const override {
    if (!has_esdf) return std::numeric_limits<double>::infinity();
    return esdfAt(p);
  }
};

static std::shared_ptr<DualGridMap> loadDualMapOccTxt(const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs) return nullptr;
  auto gm = std::make_shared<DualGridMap>();
  ifs >> gm->nx >> gm->ny >> gm->nz;
  ifs >> gm->resolution;
  ifs >> gm->origin.x() >> gm->origin.y() >> gm->origin.z();
  if (!ifs.good()) return nullptr;
  size_t total = static_cast<size_t>(gm->nx) * gm->ny * gm->nz;
  gm->occ.resize(total);
  for (size_t i = 0; i < total; ++i) {
    int v = 0; ifs >> v; gm->occ[i] = static_cast<uint8_t>(v ? 1 : 0);
    if (!ifs.good()) return nullptr;
  }
  return gm;
}

static bool loadDualMapEsdfTxt(DualGridMap &gm, const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs) return false;
  int nx, ny, nz; double res; Eigen::Vector3d org;
  ifs >> nx >> ny >> nz; ifs >> res; ifs >> org.x() >> org.y() >> org.z();
  if (!ifs.good()) return false;
  if (nx != gm.nx || ny != gm.ny || nz != gm.nz || std::abs(res - gm.resolution) > 1e-9) {
    std::cerr << "ESDF header mismatch with occupancy map." << std::endl;
    return false;
  }
  size_t total = static_cast<size_t>(nx) * ny * nz;
  gm.esdf.resize(total);
  for (size_t i = 0; i < total; ++i) {
    double v; ifs >> v; gm.esdf[i] = static_cast<float>(v);
    if (!ifs.good()) return false;
  }
  gm.has_esdf = true;
  return true;
}

static std::shared_ptr<GridMap> loadGridMapFromTxt(const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs) return nullptr;
  auto gm = std::make_shared<GridMap>();
  ifs >> gm->nx >> gm->ny >> gm->nz;
  ifs >> gm->resolution;
  ifs >> gm->origin.x() >> gm->origin.y() >> gm->origin.z();
  if (!ifs.good()) return nullptr;
  size_t total = static_cast<size_t>(gm->nx) * gm->ny * gm->nz;
  gm->data.resize(total);
  for (size_t i = 0; i < total; ++i) {
    int v = 0; ifs >> v; gm->data[i] = static_cast<uint8_t>(v ? 1 : 0);
    if (!ifs.good()) return nullptr;
  }
  return gm;
}

// Minimal NPY reader for little-endian C-order arrays (uint8, float32)
struct NpyHeader {
  std::string descr;
  bool fortran{false};
  std::vector<size_t> shape;
};

static bool parseNpyHeader(std::ifstream &ifs, NpyHeader &hdr, size_t &data_offset) {
  char magic[6];
  ifs.read(magic, 6);
  if (!ifs || std::string(magic, 6) != std::string("\x93NUMPY",6)) return false;
  uint8_t v_major=0, v_minor=0;
  ifs.read(reinterpret_cast<char*>(&v_major), 1);
  ifs.read(reinterpret_cast<char*>(&v_minor), 1);
  if (!ifs) return false;
  uint32_t header_len = 0;
  if (v_major == 1) {
    uint16_t hl16 = 0;
    ifs.read(reinterpret_cast<char*>(&hl16), 2);
    header_len = hl16;
  } else {
    ifs.read(reinterpret_cast<char*>(&header_len), 4);
  }
  if (!ifs) return false;
  std::string header(header_len, '\0');
  ifs.read(&header[0], header_len);
  if (!ifs) return false;
  data_offset = 6 + 2 + ((v_major==1)?2:4) + header_len;
  // crude parse of python dict header
  auto find_str = [&](const std::string &key)->std::string{
    auto pos = header.find("'" + key + "'");
    if (pos == std::string::npos) return "";
    pos = header.find("'", pos + key.size() + 2);
    if (pos == std::string::npos) return "";
    auto pos2 = header.find("'", pos + 1);
    if (pos2 == std::string::npos) return "";
    return header.substr(pos+1, pos2-pos-1);
  };
  auto find_bool = [&](const std::string &key)->bool{
    auto pos = header.find(key);
    if (pos == std::string::npos) return false;
    auto posT = header.find("True", pos);
    auto posF = header.find("False", pos);
    if (posT != std::string::npos && (posF==std::string::npos || posT < posF)) return true;
    return false;
  };
  auto find_shape = [&](){
    auto pos = header.find("shape");
    if (pos == std::string::npos) return;
    pos = header.find("(", pos);
    auto end = header.find(")", pos);
    if (pos == std::string::npos || end == std::string::npos) return;
    std::string inside = header.substr(pos+1, end-pos-1);
    std::stringstream ss(inside);
    hdr.shape.clear();
    while (ss.good()) {
      while (ss.good() && (ss.peek()==' '||ss.peek()==',')) ss.get();
      if (!std::isdigit(ss.peek())) { ss.get(); continue; }
      size_t v=0; ss >> v; hdr.shape.push_back(v);
    }
  };
  hdr.descr = find_str("descr");
  hdr.fortran = find_bool("fortran_order");
  find_shape();
  return !hdr.descr.empty() && !hdr.fortran && !hdr.shape.empty();
}

static bool loadNpyUint8(const std::string &path, std::vector<uint8_t> &out, std::vector<size_t> &shape) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) return false;
  NpyHeader h; size_t off=0;
  if (!parseNpyHeader(ifs, h, off)) return false;
  if (!(h.descr == "|u1" || h.descr == "<u1")) return false;
  if (h.fortran) return false;
  size_t total = 1; for (auto v: h.shape) total *= v;
  out.resize(total);
  ifs.read(reinterpret_cast<char*>(out.data()), total * sizeof(uint8_t));
  if (!ifs) return false;
  shape = h.shape;
  return true;
}

static bool loadNpyFloat32(const std::string &path, std::vector<float> &out, std::vector<size_t> &shape) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) return false;
  NpyHeader h; size_t off=0;
  if (!parseNpyHeader(ifs, h, off)) return false;
  if (!(h.descr == "<f4" || h.descr == "|f4")) return false;
  if (h.fortran) return false;
  size_t total = 1; for (auto v: h.shape) total *= v;
  out.resize(total);
  ifs.read(reinterpret_cast<char*>(out.data()), total * sizeof(float));
  if (!ifs) return false;
  shape = h.shape;
  return true;
}

struct Args {
  // inputs
  std::string map_file; // optional: direct occupancy ASCII file
  std::string map_base; // optional: base path to resolve "<base>_occ.txt" and "<base>_esdf.npz"
  std::string esdf_file; // optional: ESDF npz (currently not consumed by planner)
  std::string output_file; // optional

  // state
  Eigen::Vector3d start_pos{Eigen::Vector3d(-2,0,0)};
  Eigen::Vector3d start_vel{Eigen::Vector3d::Zero()};
  Eigen::Vector3d start_acc{Eigen::Vector3d::Zero()};
  Eigen::Vector3d goal_pos{Eigen::Vector3d(2,0,0)};
  Eigen::Vector3d goal_vel{Eigen::Vector3d::Zero()};
  Eigen::Vector3d goal_acc{Eigen::Vector3d::Zero()};

  // planner params
  PlannerConfig cfg;
  // ESDF usage
  double inflate_radius{0.0};
  // Grid resolution for .npy maps (cell size in meters)
  double grid_resolution{0.0};
};

static bool parseArgs(int argc, char** argv, Args &a) {
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto need = [&](int n){ return i + n < argc; };
    auto nextd = [&](int &idx){ return std::stod(argv[++idx]); };
    auto nexti = [&](int &idx){ return std::stoi(argv[++idx]); };
    if (k == "--map" && need(1)) { a.map_file = argv[++i]; }
    else if (k == "--map-base" && need(1)) { a.map_base = argv[++i]; }
    else if (k == "--esdf" && need(1)) { a.esdf_file = argv[++i]; }
  else if (k == "--output" && need(1)) { a.output_file = argv[++i]; }
  else if (k == "--inflate" && need(1)) { a.inflate_radius = nextd(i); }
    else if (k == "--start" && need(3)) { a.start_pos.x() = nextd(i); a.start_pos.y() = nextd(i); a.start_pos.z() = nextd(i); }
    else if (k == "--start-vel" && need(3)) { a.start_vel.x() = nextd(i); a.start_vel.y() = nextd(i); a.start_vel.z() = nextd(i); }
    else if (k == "--start-acc" && need(3)) { a.start_acc.x() = nextd(i); a.start_acc.y() = nextd(i); a.start_acc.z() = nextd(i); }
  else if (k == "--goal" && need(3)) { a.goal_pos.x() = nextd(i); a.goal_pos.y() = nextd(i); a.goal_pos.z() = nextd(i); }
  else if (k == "--goal-vel" && need(3)) { a.goal_vel.x() = nextd(i); a.goal_vel.y() = nextd(i); a.goal_vel.z() = nextd(i); }
  else if (k == "--goal-acc" && need(3)) { a.goal_acc.x() = nextd(i); a.goal_acc.y() = nextd(i); a.goal_acc.z() = nextd(i); }
    else if (k == "--goal-tol" && need(1)) { a.cfg.goal_tolerance = nextd(i); }
    else if (k == "--res" && need(1)) { a.cfg.resolution = nextd(i); }
    else if (k == "--step" && need(1)) { a.cfg.step_size = nextd(i); }
  else if (k == "--grid-res" && need(1)) { a.grid_resolution = nextd(i); }
  else if (k == "--safety-margin" && need(1)) { a.cfg.safety_margin = nextd(i); }
  else if (k == "--safety-weight" && need(1)) { a.cfg.safety_weight = nextd(i); }
    else if (k == "--map-size" && need(3)) { a.cfg.map_size.x() = nextd(i); a.cfg.map_size.y() = nextd(i); a.cfg.map_size.z() = nextd(i); }
    else if (k == "--map-origin" && need(3)) { a.cfg.map_origin.x() = nextd(i); a.cfg.map_origin.y() = nextd(i); a.cfg.map_origin.z() = nextd(i); }
    else if (k == "--max-iters" && need(1)) { a.cfg.max_iterations = nexti(i); }
    else if (k == "--time-limit" && need(1)) { a.cfg.time_limit_sec = nextd(i); }
    else if (k == "--stop-on-first" && need(1)) { int v = nexti(i); a.cfg.stop_on_first_feasible = (v != 0); }
    else if (k == "-h" || k == "--help") {
      std::cout
        << "Usage: " << argv[0] << " [--map occ.{txt|npy} | --map-base BASE [--esdf BASE_esdf.{txt|npy}]] [--output out.json]\n"
        << "  --start x y z [--start-vel vx vy vz] [--start-acc ax ay az]\n"
  << "  --goal x y z [--goal-vel vx vy vz] [--goal-acc ax ay az] [--goal-tol v] [--res v] [--step v]\n"
        << "  [--map-size sx sy sz] [--map-origin ox oy oz]\n"
  << "  [--max-iters N] [--time-limit S] [--stop-on-first 0|1]\n"
  << "  [--inflate r]  # Use ESDF to inflate obstacles by radius r (m) when ESDF is provided.\n"
        << "  [--grid-res v] # Cell size (meters) required when using .npy occupancy/ESDF maps.\n"
  << "  [--safety-margin m] [--safety-weight w] # ESDF soft safety cost inside margin m (m) with weight w.\n"
        << "\nNotes:\n"
        << "  - Occupancy ASCII format:\n"
        << "      line1: nx ny nz\n      line2: resolution\n      line3: origin_x origin_y origin_z (map center)\n"
        << "      followed by nx*ny*nz values (0/1) in C-order.\n"
        << "  - With --map-base BASE, this program will try to read BASE + '_occ.npy' or '_occ.txt'.\n"
        << "  - ESDF: provide BASE_esdf.npy (preferred) or BASE_esdf.txt; --grid-res is required with .npy.\n"
        << "    The planner uses ESDF for safety inflation when --inflate > 0.\n";
      return false;
    }
  }
  return true;
}

static void writeJsonOutput(const std::string &path, bool success,
                            const std::vector<Eigen::Matrix<double,6,3>> &coeffs,
                            const std::vector<double> &intervals,
                            double cost) {
  std::ofstream ofs(path);
  if (!ofs) return;
  ofs << "{\n";
  ofs << "  \"success\": " << (success ? "true" : "false") << ",\n";
  ofs << "  \"cost\": " << cost << ",\n";
  ofs << "  \"intervals\": [";
  for (size_t i = 0; i < intervals.size(); ++i) {
    if (i) ofs << ", ";
    ofs << intervals[i];
  }
  ofs << "],\n";
  ofs << "  \"coefficients\": [\n";
  for (size_t s = 0; s < coeffs.size(); ++s) {
    if (s) ofs << ",\n";
    ofs << "    [";
    for (int r = 0; r < 6; ++r) {
      if (r) ofs << ", ";
      ofs << "[" << coeffs[s](r,0) << ", " << coeffs[s](r,1) << ", " << coeffs[s](r,2) << "]";
    }
    ofs << "]";
  }
  ofs << "\n  ]\n";
  ofs << "}\n";
}

int main(int argc, char** argv) {
  Args args;
  if (!parseArgs(argc, argv, args)) {
    return 1;
  }

  std::shared_ptr<OccupancyMapInterface> map;
  std::shared_ptr<DualGridMap> gm;
  if (!args.map_file.empty()) {
    // Decide by extension
    if (args.map_file.size() >= 4 && args.map_file.substr(args.map_file.size()-4) == ".npy") {
      std::vector<uint8_t> occ;
      std::vector<size_t> shape;
      if (!loadNpyUint8(args.map_file, occ, shape)) {
        std::cerr << "Failed to load NPY occupancy from " << args.map_file << std::endl;
        return 2;
      }
      if (shape.size() != 3) {
        std::cerr << "NPY occupancy must be 3D (nx,ny,nz)." << std::endl;
        return 2;
      }
      if (args.grid_resolution <= 0.0) {
        std::cerr << "--grid-res is required when using .npy maps." << std::endl;
        return 2;
      }
      gm = std::make_shared<DualGridMap>();
      gm->nx = static_cast<int>(shape[0]);
      gm->ny = static_cast<int>(shape[1]);
      gm->nz = static_cast<int>(shape[2]);
      gm->resolution = args.grid_resolution;
      gm->occ = std::move(occ);
    } else {
      gm = loadDualMapOccTxt(args.map_file);
      if (!gm) {
        std::cerr << "Failed to load map from " << args.map_file << std::endl;
        return 2;
      }
    }
  } else if (!args.map_base.empty()) {
    // Prefer .npy occupancy
    std::string occ_npy = args.map_base + "_occ.npy";
    std::string occ_txt = args.map_base + "_occ.txt";
    std::ifstream t1(occ_npy), t2(occ_txt);
    if (t1.good()) {
      std::vector<uint8_t> occ;
      std::vector<size_t> shape;
      if (!loadNpyUint8(occ_npy, occ, shape)) {
        std::cerr << "Failed to load NPY occupancy from base: " << occ_npy << std::endl;
        return 2;
      }
      if (shape.size() != 3) { std::cerr << "NPY occupancy must be 3D." << std::endl; return 2; }
      if (args.grid_resolution <= 0.0) { std::cerr << "--grid-res is required when using .npy maps." << std::endl; return 2; }
      gm = std::make_shared<DualGridMap>();
      gm->nx = static_cast<int>(shape[0]);
      gm->ny = static_cast<int>(shape[1]);
      gm->nz = static_cast<int>(shape[2]);
      gm->resolution = args.grid_resolution;
      gm->occ = std::move(occ);
    } else {
      gm = loadDualMapOccTxt(occ_txt);
      if (!gm) {
        std::cerr << "Failed to load occupancy map from base: " << occ_txt << std::endl;
        return 2;
      }
    }
    // gm has been set by either npy or txt branch above
    // Try load ESDF (prefer .npy)
    std::string esdf_path = args.esdf_file;
    if (esdf_path.empty()) {
      std::string npy_guess = args.map_base + std::string("_esdf.npy");
      std::ifstream npt(npy_guess);
      if (npt.good()) esdf_path = npy_guess; else {
        std::string ascii_guess = args.map_base + std::string("_esdf.txt");
        std::ifstream ast(ascii_guess);
        if (ast.good()) esdf_path = ascii_guess;
      }
    }
    if (!esdf_path.empty()) {
      if (esdf_path.size() >= 4 && esdf_path.substr(esdf_path.size()-4) == ".npy") {
        std::vector<float> esdf;
        std::vector<size_t> shape;
        if (!loadNpyFloat32(esdf_path, esdf, shape)) {
          std::cerr << "Warning: failed to load ESDF NPY from " << esdf_path << std::endl;
        } else {
          if (shape.size() == 3 && static_cast<int>(shape[0])==gm->nx && static_cast<int>(shape[1])==gm->ny && static_cast<int>(shape[2])==gm->nz) {
            gm->esdf = std::move(esdf);
            gm->has_esdf = true;
          } else {
            std::cerr << "Warning: ESDF shape mismatch with occupancy." << std::endl;
          }
        }
      } else if (esdf_path.size() >= 4 && esdf_path.substr(esdf_path.size()-4) == ".txt") {
        if (!loadDualMapEsdfTxt(*gm, esdf_path)) {
          std::cerr << "Warning: failed to load ESDF ASCII from " << esdf_path << std::endl;
        }
      }
    }
    // sync planner map_size/map_origin from grid map if not manually overridden
    args.cfg.map_origin = gm->origin;
    args.cfg.map_size = Eigen::Vector3d(gm->nx * gm->resolution, gm->ny * gm->resolution, gm->nz * gm->resolution);
    args.cfg.resolution = std::max(args.cfg.resolution, 1e-3); // planner collision step (not cell size)
    gm->inflate_radius = args.inflate_radius;
    map = gm;
  } else {
    // Fallback empty map (no obstacles), treat outside bounds as occupied via map_size
    struct EmptyMap : public OccupancyMapInterface {
      bool hasObservation() const override { return true; }
      bool isOccupied(const Eigen::Vector3d &) const override { return false; }
    };
    map = std::make_shared<EmptyMap>();
  }

  // If --map used, also sync cfg and set inflation before planner init
  if (gm) {
    args.cfg.map_origin = gm->origin;
    args.cfg.map_size = Eigen::Vector3d(gm->nx * gm->resolution, gm->ny * gm->resolution, gm->nz * gm->resolution);
    args.cfg.resolution = std::max(args.cfg.resolution, 1e-3);
    gm->inflate_radius = args.inflate_radius;
    map = gm;
  }

  KinodynRRTStarPlanner planner;
  planner.initPlanner(args.cfg, map);

  bool ok = planner.searchTraj(args.start_pos, args.start_vel, args.start_acc, args.goal_pos, args.goal_vel, args.goal_acc);

  if (!args.output_file.empty()) {
    writeJsonOutput(args.output_file, ok, planner.getTrajCoeff(), planner.getTrajInterval(), planner.getCost());
  } else {
    std::cout << "searchTraj: " << (ok ? "OK" : "FAIL") << std::endl;
    std::cout << "cost: " << planner.getCost() << std::endl;
    std::cout << "segments: " << planner.getTrajInterval().size() << std::endl;
  }

  return ok ? 0 : 1;
}
