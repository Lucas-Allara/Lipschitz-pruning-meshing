// Random scenes for the scaling experiments and the tests.
#pragma once
#include <string>

#include "lpm/scene.h"

namespace lpm {

struct SyntheticParams {
    int n = 100;              // number of primitives
    double radius = 0.05;     // primitive size (sphere radius; the other shapes scale with it)
    double k = 0.0;           // blending radius of every operator
    unsigned seed = 1;
    double extent = 1.0;      // centers are uniform in [-extent, extent]^3
    std::string shapes = "sphere";  // "sphere" or "mixed" (sphere, box, cylinder, cone with random rotations)
    std::string ops = "union";      // "union" or "mixed" (union, difference, intersection)
    bool chain = false;       // left-deep chain instead of a balanced tree
    bool morton = false;      // sort the primitives along a Morton curve before building the tree
};

Scene make_synthetic(const SyntheticParams& p);

// Writes a scene in the JSON format of the official repository; false when the file cannot be
// written.
[[nodiscard]] bool write_scene_json(const Scene& s, const std::string& path);

}  // namespace lpm
