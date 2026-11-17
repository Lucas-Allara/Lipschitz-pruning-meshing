cmake_minimum_required(VERSION 3.16)
project(lpm CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)
find_package(Eigen3 QUIET NO_MODULE)

add_library(lpm src/scene.cpp src/scene_writer.cpp src/synthetic.cpp)
target_include_directories(lpm PUBLIC include external)
if(TARGET Eigen3::Eigen)
  target_link_libraries(lpm PUBLIC Eigen3::Eigen)
else()
  target_include_directories(lpm PUBLIC /usr/include/eigen3)
endif()
target_link_libraries(lpm PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(lpm PUBLIC -O3 -march=native -Wall -Wextra)

add_executable(lpm_mesh apps/lpm_mesh.cpp)
target_link_libraries(lpm_mesh lpm)
add_executable(lpm_synth apps/lpm_synth.cpp)
target_link_libraries(lpm_synth lpm)
add_executable(lpm_anim apps/lpm_anim.cpp)
target_link_libraries(lpm_anim lpm)
add_executable(lpm_server apps/lpm_server.cpp)
target_link_libraries(lpm_server lpm)

# Octree build on the GPU (Vulkan compute). The shader is compiled to SPIR-V and embedded.
find_package(Vulkan QUIET)
find_program(GLSLC glslc)
option(LPM_WITH_VULKAN "Build the GPU octree (needs Vulkan and glslc)" ON)
if(LPM_WITH_VULKAN AND Vulkan_FOUND AND GLSLC)
  set(LPM_SPV)
  file(GLOB LPM_GLSL ${CMAKE_SOURCE_DIR}/shaders/*.glsl)
  # name|source|extra glslc flags; dc.comp is compiled once per stage.
  set(LPM_SHADERS prune|prune.comp| prune_wide|prune_wide.comp| prune_interval|prune.comp|-DINTERVAL
                  prune_wide_interval|prune_wide.comp|-DINTERVAL scan|scan.comp| scan_add|scan_add.comp| grid|grid.comp|
                  dyn_prune|dyn_prune.comp| dyn_prune_wide|dyn_prune.comp|-DWIDE dyn_prune_value|dyn_prune.comp|-DVALUE
                  dyn_args|dyn_args.comp| probe|probe.comp|)
  foreach(st 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21)
    list(APPEND LPM_SHADERS dc_${st}|dc.comp|-DSTAGE=${st})
  endforeach()
  foreach(entry ${LPM_SHADERS})
    string(REPLACE "|" ";" parts "${entry}")
    list(GET parts 0 name)
    list(GET parts 1 src)
    list(LENGTH parts np)
    set(flags)
    if(np GREATER 2)
      list(GET parts 2 flags)
    endif()
    set(out ${CMAKE_BINARY_DIR}/shaders/${name}.comp.inc)
    add_custom_command(OUTPUT ${out}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
      COMMAND ${GLSLC} -O --target-env=vulkan1.2 ${flags} -mfmt=c ${CMAKE_SOURCE_DIR}/shaders/${src} -o ${out}
      DEPENDS ${CMAKE_SOURCE_DIR}/shaders/${src} ${LPM_GLSL}
      COMMENT "Compiling shaders/${src} ${flags}")
    list(APPEND LPM_SPV ${out})
  endforeach()
  add_library(lpm_vulkan src/gpu.cpp ${LPM_SPV})
  target_include_directories(lpm_vulkan PRIVATE ${CMAKE_BINARY_DIR}/shaders)
  target_link_libraries(lpm_vulkan PUBLIC lpm Vulkan::Vulkan)
  # Vulkan structures are initialized as T x{sType}, which leaves the other fields zero on purpose.
  target_compile_options(lpm_vulkan PRIVATE -Wno-missing-field-initializers)
  add_executable(lpm_gpu apps/lpm_gpu.cpp)
  target_link_libraries(lpm_gpu lpm_vulkan)
  target_compile_definitions(lpm_mesh PRIVATE LPM_HAVE_GPU)
  target_link_libraries(lpm_mesh lpm_vulkan)
  foreach(app lpm_anim lpm_server)
    target_compile_definitions(${app} PRIVATE LPM_HAVE_GPU)
    target_link_libraries(${app} lpm_vulkan)
  endforeach()
  set(LPM_GPU_TESTS ON)
  # Viewer that draws the mesh from the buffers of the dynamic update (needs SDL2 for the window).
  find_package(SDL2 QUIET)
  if(SDL2_FOUND)
    set(LPM_VIEW_SPV)
    foreach(sh view_mesh.vert view_mesh.frag)
      string(REPLACE "." "_" name ${sh})
      set(out ${CMAKE_BINARY_DIR}/shaders/${name}.inc)
      add_custom_command(OUTPUT ${out}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
        COMMAND ${GLSLC} -O --target-env=vulkan1.3 -mfmt=c ${CMAKE_SOURCE_DIR}/shaders/${sh} -o ${out}
        DEPENDS ${CMAKE_SOURCE_DIR}/shaders/${sh} ${LPM_GLSL}
        COMMENT "Compiling shaders/${sh}")
      list(APPEND LPM_VIEW_SPV ${out})
    endforeach()
    add_executable(lpm_view apps/lpm_view.cpp ${LPM_VIEW_SPV})
    target_include_directories(lpm_view PRIVATE ${CMAKE_BINARY_DIR}/shaders)
    target_link_libraries(lpm_view lpm_vulkan SDL2::SDL2)
    target_compile_options(lpm_view PRIVATE -Wno-missing-field-initializers)
  else()
    message(STATUS "SDL2 not found: lpm_view is not built")
  endif()
elseif(LPM_WITH_VULKAN)
  message(STATUS "Vulkan or glslc not found: the GPU octree is not built")
endif()

enable_testing()
foreach(t test_sdf test_pruning test_mesh)
  add_executable(${t} tests/${t}.cpp)
  target_link_libraries(${t} lpm)
endforeach()
add_test(NAME sdf COMMAND test_sdf)
add_test(NAME pruning COMMAND test_pruning)
add_test(NAME mesh COMMAND test_mesh)
add_test(NAME sdf_official COMMAND test_sdf ${CMAKE_SOURCE_DIR}/scenes/trees.json ${CMAKE_SOURCE_DIR}/scenes/molecule.json)
add_test(NAME dynamic COMMAND lpm_anim --scene ${CMAKE_SOURCE_DIR}/scenes/synthetic/spheres_n100_k0.05.json --frames 6 --depth 6 --check --stats /dev/null)
add_test(NAME patch COMMAND lpm_anim --scene ${CMAKE_SOURCE_DIR}/scenes/synthetic/spheres_n100_k0.05.json --frames 6 --depth 6 --method patch --check --stats /dev/null)
add_test(NAME pruning_official COMMAND test_pruning ${CMAKE_SOURCE_DIR}/scenes/trees.json ${CMAKE_SOURCE_DIR}/scenes/molecule.json ${CMAKE_SOURCE_DIR}/scenes/monument.json)
# A scene or an output that cannot be opened is reported, with no crash and no silent success.
add_test(NAME bad_scene COMMAND lpm_mesh --scene ${CMAKE_SOURCE_DIR}/scenes/no_such_scene.json --depth 4)
set_tests_properties(bad_scene PROPERTIES PASS_REGULAR_EXPRESSION "cannot open scene")
add_test(NAME bad_output COMMAND lpm_mesh --scene ${CMAKE_SOURCE_DIR}/scenes/synthetic/spheres_n10_k0.json --depth 4 --stats ${CMAKE_BINARY_DIR}/no_such_directory/stats.json)
set_tests_properties(bad_output PROPERTIES PASS_REGULAR_EXPRESSION "cannot write")
if(LPM_GPU_TESTS)
  add_test(NAME gpu_octree COMMAND lpm_gpu --scene ${CMAKE_SOURCE_DIR}/scenes/synthetic/spheres_n100_k0.05.json --depth 6 --repeat 1 --mesh --stats /dev/null)
  add_test(NAME gpu_octree_official COMMAND lpm_gpu --scene ${CMAKE_SOURCE_DIR}/scenes/trees.json --depth 7 --repeat 1 --mesh --stats /dev/null)
  # Dual contouring on the GPU: same topology as the CPU, and the pruned mesh identical to the full-tree one.
  add_test(NAME gpu_mesh COMMAND lpm_gpu --scene ${CMAKE_SOURCE_DIR}/scenes/synthetic/spheres_n100_k0.05.json --depth 6 --repeat 1 --gpu-mesh --full --stats /dev/null)
  add_test(NAME gpu_mesh_official COMMAND lpm_gpu --scene ${CMAKE_SOURCE_DIR}/scenes/trees.json --depth 7 --repeat 1 --gpu-mesh --full --stats /dev/null)
  add_test(NAME gpu_mesh_monument COMMAND lpm_gpu --scene ${CMAKE_SOURCE_DIR}/scenes/monument.json --depth 7 --repeat 1 --gpu-mesh --stats /dev/null)
  # Interval pruning: the leaves and trees of the CPU interval octree, and the mesh of the full tree
  # (Monument has trees above kWide, which go through prune_wide.comp).
  add_test(NAME gpu_interval COMMAND lpm_gpu --scene ${CMAKE_SOURCE_DIR}/scenes/synthetic/spheres_n100_k0.05.json --depth 6 --repeat 1 --interval --mesh --gpu-mesh --full --stats /dev/null)
  add_test(NAME gpu_interval_official COMMAND lpm_gpu --scene ${CMAKE_SOURCE_DIR}/scenes/monument.json --depth 7 --repeat 1 --interval --mesh --stats /dev/null)
  # Coarse leaves on the GPU: the coarse leaves, vertices and triangles of adaptive_mesh with top_down, and a closed mesh.
  # The CAD scenes of scenes/inversecsg are not redistributed (see its README): the test runs where they are.
  if(EXISTS ${CMAKE_SOURCE_DIR}/scenes/inversecsg/lock.json)
    add_test(NAME gpu_coarse COMMAND lpm_gpu --scene ${CMAKE_SOURCE_DIR}/scenes/inversecsg/lock.json --depth 8 --repeat 1 --coarse --adaptive-tol 0.005 --stats /dev/null)
  endif()
  add_test(NAME gpu_coarse_official COMMAND lpm_gpu --scene ${CMAKE_SOURCE_DIR}/scenes/trees.json --depth 7 --repeat 1 --coarse --stats /dev/null)
  # The octree of dense sampling (cull_factor > 1, lpm_mesh --gpu --dense-trees): the trees and the level statistics of the CPU one.
  add_test(NAME gpu_dense COMMAND lpm_gpu --scene ${CMAKE_SOURCE_DIR}/scenes/synthetic/spheres_n100_k0.05.json --depth 6 --repeat 1 --dense --stats /dev/null)
  add_test(NAME gpu_dense_official COMMAND lpm_gpu --scene ${CMAKE_SOURCE_DIR}/scenes/trees.json --depth 7 --repeat 1 --dense --stats /dev/null)
  # Warm runs with the mesh kept on the GPU (lpm_mesh --gpu --repeat) do the work of the first run, also with coarse leaves.
  add_test(NAME gpu_repeat COMMAND lpm_mesh --gpu --scene ${CMAKE_SOURCE_DIR}/scenes/synthetic/spheres_n100_k0.05.json --depth 6 --repeat 2 --stats /dev/null)
  add_test(NAME gpu_repeat_coarse COMMAND lpm_mesh --gpu --scene ${CMAKE_SOURCE_DIR}/scenes/trees.json --depth 7 --adaptive-topdown --repeat 2 --stats /dev/null)
  # The dynamic update on the GPU: the same triangles as the GPU mesh from scratch and the same
  # faces as the CPU patch, frame by frame.
  add_test(NAME gpu_patch COMMAND lpm_anim --scene ${CMAKE_SOURCE_DIR}/scenes/synthetic/spheres_n100_k0.05.json --frames 6 --depth 6 --method gpu-patch --check --stats /dev/null)
  add_test(NAME gpu_patch_official COMMAND lpm_anim --scene ${CMAKE_SOURCE_DIR}/scenes/monument.json --frames 6 --depth 7 --method gpu-patch --check --stats /dev/null)
  # The frames kept on the GPU, as lpm_view draws them: the number of triangles of both references.
  add_test(NAME gpu_patch_keep COMMAND lpm_anim --scene ${CMAKE_SOURCE_DIR}/scenes/synthetic/spheres_n100_k0.05.json --frames 6 --depth 6 --method gpu-patch --keep-on-gpu --check --stats /dev/null)
  # Marching cubes on the leaves: the patch has the triangles of the marching cubes mesh from scratch.
  add_test(NAME gpu_patch_mc COMMAND lpm_anim --scene ${CMAKE_SOURCE_DIR}/scenes/synthetic/spheres_n100_k0.05.json --frames 6 --depth 6 --method gpu-patch --mc --check --stats /dev/null)
  add_test(NAME gpu_patch_mc_official COMMAND lpm_anim --scene ${CMAKE_SOURCE_DIR}/scenes/monument.json --frames 6 --depth 7 --method gpu-patch --mc --check --stats /dev/null)
  # Coarse leaves in the patch, which no mesh from scratch reproduces: every frame closed and consistently oriented.
  add_test(NAME gpu_patch_coarse_official COMMAND lpm_anim --scene ${CMAKE_SOURCE_DIR}/scenes/trees.json --frames 6 --depth 7 --method gpu-patch --coarse --check --stats /dev/null)
  if(EXISTS ${CMAKE_SOURCE_DIR}/scenes/inversecsg/lock.json)
    add_test(NAME gpu_patch_coarse COMMAND lpm_anim --scene ${CMAKE_SOURCE_DIR}/scenes/inversecsg/lock.json --frames 6 --depth 8 --method gpu-patch --coarse --check --stats /dev/null)
  endif()
endif()

# Optional comparison against libfive (interval arithmetic octree + dual contouring).
option(LPM_WITH_LIBFIVE "Build the libfive comparison" OFF)
if(LPM_WITH_LIBFIVE)
  # LIBFIVE_INCLUDE_DIR: libfive/include of the source tree; LIBFIVE_LIBRARY: the built
  # libfive.so; LIBFIVE_EXTRA_INCLUDE: Boost headers if they are not installed system-wide.
  add_executable(lpm_libfive apps/lpm_libfive.cpp)
  target_include_directories(lpm_libfive PRIVATE ${LIBFIVE_INCLUDE_DIR} ${LIBFIVE_EXTRA_INCLUDE})
  target_link_libraries(lpm_libfive lpm ${LIBFIVE_LIBRARY})
endif()
