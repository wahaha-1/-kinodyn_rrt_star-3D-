# Kinodynamic RRT* (standalone)

A small CMake project wrapping the planner into a reusable static library and a tiny executable for quick sanity checks.

## Layout
- include/
  - kinodyn_rrt_star/kinodyn_rrt_star.h — public API
  - kdtree/kdtree.h — C header shim (reusing original header)
- src/
  - kinodyn_rrt_star.cpp — planner implementation (reusing original file for now)
  - kdtree.cpp — KD-tree implementation (reusing original file)
- app/
  - main.cpp — minimal runner for smoke tests

## Build
```bash
mkdir -p build && cd build
cmake ..
cmake --build . -j
./bin/kino_rrt_planner
```

Eigen3 is required (system package provides headers). If CMake cannot find Eigen, install `libeigen3-dev` on Debian/Ubuntu, or set `-DEigen3_DIR` accordingly.

## Notes
- During migration, the implementation files include the original sources from `src/my_simple_planner/...` to avoid duplication. You can later move the full code here and remove the relative includes.
- The library target is `kinodyn_rrt_star`; the runner `kino_rrt_planner` links it.
