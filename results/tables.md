# Results

### Official scenes, depth 8 (256^3 finest cells), 32 threads

| scene | nodes | method | total s | octree s | DC s | primitive evals | avg tree (leaves) | max tree | pruned trees MB | peak RSS MB | triangles | mesh identical to full |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| trees | 371 | octree + DC, full tree | 0.900 | 0.256 | 0.570 | 517.8 M | 371.00 | 371 | 0.0 | 130 | 428612 | yes |
|  | 371 | octree + DC, Lipschitz pruning | 0.174 | 0.047 | 0.067 | 12.5 M | 5.77 | 67 | 36.4 | 192 | 428612 | yes |
|  | 371 | octree + DC, interval pruning | 0.234 | 0.097 | 0.074 | 18.9 M | 11.28 | 73 | 72.6 | 229 | 428612 | yes |
|  | 371 | octree + DC, Lipschitz pruning + interval culling | 0.175 | 0.052 | 0.061 | 14.0 M | 7.08 | 67 | 34.8 | 159 | 428612 | yes |
|  |  | libfive (interval, DC) | 1.212 | - | - | - | - | - | - | 1473 | 305260 | see Hausdorff |
| monument | 6323 | octree + DC, full tree | 37.5 | 9.241 | 28.1 | 25.14 G | 6323.00 | 6323 | 0.0 | 375 | 1179256 | yes |
|  | 6323 | octree + DC, Lipschitz pruning | 0.570 | 0.187 | 0.203 | 47.3 M | 5.64 | 91 | 148.9 | 586 | 1179256 | yes |
|  | 6323 | octree + DC, interval pruning | 0.560 | 0.208 | 0.179 | 44.9 M | 6.92 | 77 | 144.2 | 551 | 1179256 | yes |
|  | 6323 | octree + DC, Lipschitz pruning + interval culling | 0.524 | 0.166 | 0.170 | 48.4 M | 6.52 | 83 | 122.0 | 480 | 1179256 | yes |
| molecule | 2001 | octree + DC, full tree | 3.572 | 0.801 | 2.687 | 3.72 G | 2001.00 | 2001 | 0.0 | 142 | 504440 | yes |
|  | 2001 | octree + DC, Lipschitz pruning | 0.291 | 0.105 | 0.082 | 31.8 M | 7.50 | 79 | 92.6 | 294 | 504440 | yes |
|  | 2001 | octree + DC, interval pruning | 0.392 | 0.217 | 0.090 | 52.3 M | 9.52 | 121 | 163.9 | 432 | 504440 | yes |
|  | 2001 | octree + DC, Lipschitz pruning + interval culling | 0.267 | 0.109 | 0.078 | 35.8 M | 7.62 | 79 | 90.9 | 285 | 504440 | yes |
|  |  | libfive (interval, DC) | 21.2 | - | - | - | - | - | - | 3736 | 504440 | see Hausdorff |

### Speedups of Lipschitz pruning (depth 8)

| scene | primitive evals, full / Lipschitz | time, full / Lipschitz | nodes / avg leaf tree (H2 estimate) | leaf cells with one primitive | primitive evals, interval / Lipschitz | time, interval / Lipschitz |
|---|---|---|---|---|---|---|
| trees | 41x | 5.2x | 64x | 51 % | 1.51x | 1.34x |
| monument | 532x | 65.9x | 1120x | 37 % | 0.95x | 0.98x |
| molecule | 117x | 12.3x | 267x | 13 % | 1.64x | 1.35x |

### Scaling: random spheres of radius 0.05 in [-1, 1]^3, depth 8

| spheres | k | method | total s | primitive evals | avg leaf tree | triangles |
|---|---|---|---|---|---|---|
| 10 | 0 | full | 0.008 | 636.0 k | 19.00 | 8832 |
| 100 | 0 | full | 0.091 | 78.8 M | 199.00 | 111388 |
| 1000 | 0 | full | 7.066 | 7.55 G | 1999.00 | 1085540 |
| 10 | 0 | lipschitz | 0.008 | 66.2 k | 1.00 | 8832 |
| 100 | 0 | lipschitz | 0.028 | 983.7 k | 1.02 | 111388 |
| 1000 | 0 | lipschitz | 0.235 | 13.9 M | 1.17 | 1085540 |
| 10000 | 0 | lipschitz | 1.618 | 181.1 M | 3.06 | 5191052 |
| 10 | 0 | interval | 0.007 | 60.7 k | 1.00 | 8832 |
| 100 | 0 | interval | 0.028 | 853.1 k | 1.01 | 111388 |
| 1000 | 0 | interval | 0.213 | 11.2 M | 1.14 | 1085540 |
| 10000 | 0 | interval | 1.494 | 132.5 M | 2.67 | 5191052 |
| 10 | 0.05 | full | 0.008 | 590.6 k | 19.00 | 8152 |
| 100 | 0.05 | full | 0.089 | 73.3 M | 199.00 | 102772 |
| 1000 | 0.05 | full | 7.850 | 7.16 G | 1999.00 | 995224 |
| 10 | 0.05 | lipschitz | 0.007 | 62.0 k | 1.00 | 8152 |
| 100 | 0.05 | lipschitz | 0.029 | 1.1 M | 1.18 | 102772 |
| 1000 | 0.05 | lipschitz | 0.279 | 27.5 M | 3.54 | 995224 |
| 10000 | 0.05 | lipschitz | 4.579 | 1.07 G | 73.16 | 2456856 |
| 10 | 0.05 | interval | 0.007 | 56.9 k | 1.00 | 8152 |
| 100 | 0.05 | interval | 0.028 | 1.0 M | 1.27 | 102772 |
| 1000 | 0.05 | interval | 0.356 | 35.4 M | 4.48 | 995224 |
| 10000 | 0.05 | interval | 8.780 | 1.51 G | 85.25 | 2456856 |

### Blending radius sweep: 1000 spheres of radius r = 0.05, depth 8

| k | k / r | method | total s | primitive evals | avg leaf tree | max leaf tree |
|---|---|---|---|---|---|---|
| 0 | 0.00 | lipschitz | 0.448 | 16.2 M | 1.17 | 9 |
| 0.0125 | 0.25 | lipschitz | 0.484 | 17.9 M | 1.40 | 13 |
| 0.025 | 0.50 | lipschitz | 0.466 | 20.6 M | 1.79 | 17 |
| 0.05 | 1.00 | lipschitz | 0.454 | 31.6 M | 3.54 | 27 |
| 0.1 | 2.00 | lipschitz | 0.924 | 147.8 M | 22.70 | 101 |
| 0.2 | 4.00 | lipschitz | 2.356 | 632.2 M | 145.83 | 525 |
| 0.4 | 8.00 | lipschitz | 3.149 | 1.20 G | 471.84 | 817 |
| 0 | 0.00 | interval | 0.438 | 13.4 M | 1.14 | 9 |
| 0.0125 | 0.25 | interval | 0.475 | 15.8 M | 1.43 | 13 |
| 0.025 | 0.50 | interval | 0.461 | 19.8 M | 1.94 | 19 |
| 0.05 | 1.00 | interval | 0.617 | 39.6 M | 4.48 | 33 |
| 0.1 | 2.00 | interval | 2.156 | 303.5 M | 33.20 | 135 |
| 0.2 | 4.00 | interval | 15.8 | 1.85 G | 176.15 | 559 |
| 0 | 0.00 | full | 9.106 | 9.73 G | - | - |
| 0.4 | 8.00 | full | 5.281 | 5.04 G | - | - |

### Adverse case: 5000 spheres of radius 0.05 packed in [-0.25, 0.25]^3, depth 8

| k | method | total s | primitive evals | avg leaf tree | max leaf tree | single-primitive leaves | triangles |
|---|---|---|---|---|---|---|---|
| 0 | full | 16.8 | 16.44 G | 9999.0 | 9999 | 0 % | 498856 |
| 0 | lipschitz | 0.295 | 38.7 M | 2.9 | 29 | 53 % | 498856 |
| 0 | interval | 0.204 | 25.7 M | 2.5 | 27 | 58 % | 498856 |
| 0.05 | full | 14.8 | 12.82 G | 9999.0 | 9999 | 0 % | 367020 |
| 0.05 | lipschitz | 2.671 | 789.5 M | 535.9 | 1199 | 0 % | 367020 |

### Depth sweep (32 threads)

| depth | cells per axis | Trees full s | Trees Lipschitz s | Trees avg leaf tree | Trees primitive evals | Monument Lipschitz s | Monument avg leaf tree | Monument triangles |
|---|---|---|---|---|---|---|---|---|
| 5 | 32 | 0.142 | 0.116 | 32.26 | 1.1 M | 0.274 | 126.01 | 8996 |
| 6 | 64 | 0.194 | 0.081 | 18.59 | 2.6 M | 0.254 | 36.57 | 60684 |
| 7 | 128 | 0.366 | 0.095 | 9.49 | 5.9 M | 0.390 | 12.08 | 271752 |
| 8 | 256 | 1.290 | 0.270 | 5.77 | 15.6 M | 0.847 | 5.64 | 1179256 |
| 9 | 512 | - | 0.942 | 4.63 | 48.1 M | 2.862 | 3.56 | 4890584 |

### Pruning from the parent tree versus from the full tree at every cell

| scene | octree s, inherited | octree s, from full | octree primitive evals, inherited | from full | avg leaf tree, inherited | from full |
|---|---|---|---|---|---|---|
| trees | 0.047 | 0.387 | 5.5 M | 145.1 M | 5.77 | 5.77 |
| monument | 0.187 | 13.8 | 30.1 M | 7.24 G | 5.64 | 5.64 |
| molecule | 0.105 | 1.210 | 22.1 M | 858.0 M | 7.50 | 7.50 |

### Analytic single-sphere leaves (crossing and normal without evaluations)

| scene | DC primitive evals | with analytic leaves | analytic crossings | DC s | DC s, analytic |
|---|---|---|---|---|---|
| molecule | 13.4 M | 12.7 M | 55898/252220 | 0.217 | 0.220 |
| spheres_n1000_k0 | 8.4 M | 1.8 M | 519998/542770 | 0.368 | 0.357 |
| spheres_n1000_k0.05 | 14.6 M | 11.2 M | 268699/497612 | 0.358 | 0.358 |

### Tree shape: balanced versus chain, and mixed primitives and operators (1000 primitives)

| scene | tree height | method | total s | primitive evals | avg leaf tree | max leaf tree |
|---|---|---|---|---|---|---|
| spheres_n1000_k0 | 11 | lipschitz | 0.553 | 16.2 M | 1.17 | 9 |
| spheres_n1000_k0 | 11 | interval | 0.532 | 13.4 M | 1.14 | 9 |
| chain_n1000_k0 | 1000 | lipschitz | 0.582 | 12.8 M | 1.17 | 9 |
| chain_n1000_k0 | 1000 | interval | 0.549 | 11.4 M | 1.14 | 9 |
| mixed_n1000_k0.025 | 11 | lipschitz | 0.227 | 2.8 M | 1.09 | 5 |
| mixed_n1000_k0.025 | 11 | interval | 0.244 | 4.4 M | 1.16 | 5 |

### Threads

| scene | Lipschitz, 1 thread s | Lipschitz, 32 threads s | parallel speedup | full tree, 1 thread s |
|---|---|---|---|---|
| trees | 0.632 | 0.174 | 3.6x | 7.099 |
| monument | 2.185 | 0.570 | 3.8x | - |
| molecule | 0.954 | 0.291 | 3.3x | - |

### Dense 257^3 discretization + marching cubes versus the sparse octree (depth 8)

| scene | pipeline | sampling / DC s | pruning s | primitive evals | pruned trees MB | peak RSS MB | MC s | triangles |
|---|---|---|---|---|---|---|---|---|
| trees | dense grid, full tree | 4.487 | 0.000 | 3.16 G | 0.0 | 69 | 0.194 | 429766 |
| trees | dense grid, Lipschitz pruning | 0.132 | 0.197 | 14.8 M | 82.6 | 337 | 0.185 | 429766 |
| trees | octree + DC, Lipschitz pruning | 0.067 | 0.047 | 12.5 M | 36.4 | 192 | - | 428612 |
| monument | dense grid, full tree | 85.3 | 0.000 | 53.67 G | 0.0 | 84 | 0.303 | 1181902 |
| monument | dense grid, Lipschitz pruning | 0.241 | 0.541 | 63.8 M | 294.2 | 1131 | 0.320 | 1181902 |
| monument | octree + DC, Lipschitz pruning | 0.203 | 0.187 | 47.3 M | 148.9 | 586 | - | 1179256 |
| molecule | dense grid, full tree | 17.8 | 0.000 | 16.99 G | 0.0 | 71 | 0.212 | 504582 |
| molecule | dense grid, Lipschitz pruning | 0.122 | 0.279 | 50.4 M | 198.0 | 611 | 0.198 | 504582 |
| molecule | octree + DC, Lipschitz pruning | 0.082 | 0.105 | 31.8 M | 92.6 | 294 | - | 504440 |

### Distances between meshes (20 000 surface samples plus the vertices of each mesh; the closest-point routine of trimesh has a noise floor of about 1e-6 to 1e-4 on the sliver triangles of marching cubes, so the mean is the reliable figure; the DC meshes of the three oracles are byte-identical). Closedness of the DC meshes: trees: 0 boundary edges, 0 non-manifold edges of 428612 triangles; monument: 0 boundary edges, 19 non-manifold edges of 1179256 triangles; molecule: 0 boundary edges, 3 non-manifold edges of 504440 triangles.

| scene | compared with | Hausdorff | mean | vertices A | vertices B | cell size |
|---|---|---|---|---|---|---|
| trees | libfive | 3.68e-03 | 1.44e-06 | 214312 | 152636 | 9.14e-03 |
| trees | dense MC (full tree) | 8.10e-03 | 9.88e-05 | 214312 | 214883 | 9.14e-03 |
| trees | dense MC (pruned) | 0 (byte-identical) | 0 | - | - | 9.14e-03 |
| monument | dense MC (full tree) | 1.47e-02 | 5.57e-04 | 588774 | 589971 | 1.09e-02 |
| monument | dense MC (pruned) | 0 (byte-identical) | 0 | - | - | 1.09e-02 |
| molecule | libfive | 2.49e-03 | 1.53e-05 | 252152 | 252152 | 8.59e-03 |
| molecule | dense MC (full tree) | 6.00e-03 | 3.68e-04 | 252152 | 252221 | 8.59e-03 |
| molecule | dense MC (pruned) | 0 (byte-identical) | 0 | - | - | 8.59e-03 |

### libfive (interval arithmetic octree + dual contouring, 32 workers, depth 8; its Tree::optimized() pass is skipped because it unfolds shared subtrees and is exponential in the height of blended trees)

| scene | libfive render s | octree + DC Lipschitz s | libfive triangles | libfive peak RSS MB |
|---|---|---|---|---|
| trees | 1.212 | 0.174 | 305260 | 1473 |
| monument | failed: memory > 24 GB | 0.570 | - | > 24000 |
| molecule | 21.2 | 0.291 | 504440 | 3736 |
| spheres_n100_k0 | 0.147 | 0.028 | 111388 | 270 |
| spheres_n1000_k0 | 1.316 | 0.235 | 1085540 | 2623 |
| spheres_n1000_k0.05 | 2.794 | 0.279 | 995224 | 3964 |
| spheres_n10000_k0 | failed: memory > 24 GB | 1.618 | - | > 24000 |

### A sphere of radius 0.2 moving through the scene (smooth union, k = 0.1), re-meshed every frame at depth 8

| scene | method | frames | mean s/frame | max s/frame | octree s | DC s | leaf cells (patch: region) | static leaves (patch: quads kept) | vertices reused | primitive evals/frame | setup s | identical to scratch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| trees | full | 48 | 1.138 | 1.317 | 0.254 | 0.877 | 335297 | - | - | 698.9 M | - | - |
| trees | scratch | 48 | 0.227 | 0.300 | 0.063 | 0.156 | 335297 | - | - | 15.9 M | - | - |
| trees | incremental | 48 | 0.267 | 0.341 | 0.091 | 0.168 | 335297 | 95 % | 62 % | 928.6 k | 0.457 | yes |
| trees | patch | 48 | 0.022 | 0.029 | 0.003 | 0.008 | 17614 | 94 % | - | 928.6 k | 0.382 | yes |
| monument | full | 4 | 43.4 | 45.7 | 9.595 | 33.7 | 1029668 | - | - | 32.99 G | - | - |
| monument | scratch | 48 | 0.711 | 0.757 | 0.206 | 0.469 | 1025255 | - | - | 55.3 M | - | - |
| monument | incremental | 48 | 0.669 | 0.812 | 0.165 | 0.472 | 1025255 | 97 % | 55 % | 3.1 M | 1.057 | yes |
| monument | patch | 48 | 0.056 | 0.077 | 0.014 | 0.018 | 37523 | 96 % | - | 3.1 M | 1.161 | yes |

### Sharp features: dihedral angles between adjacent triangles (dual contouring keeps the creases of the boxes; marching cubes rounds them)

| scene | mesh | interior edges | angle > 30 deg | angle > 60 deg | 99th percentile (deg) | max (deg) |
|---|---|---|---|---|---|---|
| trees | octree + DC | 642918 | 0.43 % | 0.348 % | 17.3 | 180.0 |
| trees | dense MC | 644649 | 0.93 % | 0.422 % | 27.2 | 160.2 |
| monument | octree + DC | 1768846 | 8.98 % | 5.369 % | 118.3 | 180.0 |
| monument | dense MC | 1772853 | 7.73 % | 2.462 % | 81.0 | 174.4 |
| molecule | octree + DC | 756654 | 2.79 % | 0.666 % | 47.2 | 180.0 |
| molecule | dense MC | 756873 | 2.25 % | 0.135 % | 37.8 | 166.1 |

### Adaptive contouring: nodes collapsed bottom-up where the children's quadric fits one vertex (tolerance relative to the cell size) and the topology is kept

| scene | collapse rule | fine leaves | leaves | leaves saved | vertices | triangles | uniform triangles | total s | uniform s | Hausdorff / cell | mean / cell | boundary edges | non-manifold edges |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| trees | single-primitive cells, tol 0.02 | 329755 | 250856 | 24 % | 166627 | 333242 | 428612 | 0.917 | 0.174 | 0.04 | 0.0001 | 0 | 0 |
|  | any cell, tol 0.02 | 329755 | 220436 | 33 % | 144819 | 289626 | 428612 | 0.909 | 0.174 | 0.16 | 0.0010 | 0 | 0 |
|  | any cell, tol 0.1 | 329755 | 205437 | 38 % | 134657 | 269302 | 428612 | 0.920 | 0.174 | 0.26 | 0.0025 | 0 | 0 |
| monument | single-primitive cells, tol 0.02 | 1020515 | 911131 | 11 % | 541651 | 1085010 | 1179256 | 2.870 | 0.570 | 0.02 | 0.0000 | 0 | 19 |
|  | any cell, tol 0.02 | 1020515 | 775131 | 24 % | 476292 | 954292 | 1179256 | 2.946 | 0.570 | 0.91 | 0.0003 | 0 | 19 |
|  | any cell, tol 0.1 | 1020515 | 709286 | 30 % | 432066 | 865840 | 1179256 | 3.013 | 0.570 | 0.91 | 0.0023 | 0 | 19 |
| molecule | single-primitive cells, tol 0.02 | 358333 | 358333 | 0 % | 252152 | 504440 | 504440 | 0.987 | 0.291 | 0.00 | 0.0000 | 0 | 3 |
|  | any cell, tol 0.02 | 358333 | 355000 | 1 % | 251703 | 503542 | 504440 | 1.108 | 0.291 | 0.15 | 0.0000 | 0 | 3 |
|  | any cell, tol 0.1 | 358333 | 255699 | 29 % | 180272 | 360680 | 504440 | 1.123 | 0.291 | 0.40 | 0.0122 | 0 | 3 |

### Primitives sorted along a Morton curve before building the balanced tree (alternative D of the plan)

| scene | random order s | Morton order s | evals, random | evals, Morton | avg leaf tree, random | Morton | mesh |
|---|---|---|---|---|---|---|---|
| fluid_n5000_k0 | 0.567 | 0.501 | 40.3 M | 40.5 M | 2.9 | 2.8 | identical (min is associative) |
| fluid_n5000_k0.05 | 2.762 | 2.351 | 1.00 G | 893.2 M | 535.9 | 468.5 | different surface (smooth union is not associative) |
| spheres_n10000_k0 | 4.369 | 4.247 | 197.3 M | 188.2 M | 3.1 | 3.0 | identical (min is associative) |
| spheres_n10000_k0.05 | 5.822 | 4.867 | 1.25 G | 992.3 M | 73.2 | 53.6 | different surface (smooth union is not associative) |
