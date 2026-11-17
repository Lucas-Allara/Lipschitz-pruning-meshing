# InverseCSG models

The 35 CSG programs of the `example` folder of github.com/yijiangh/InverseCSG (the code of
*InverseCSG: Automatic Conversion of 3D Models to CSG Trees*, Du et al., SIGGRAPH Asia 2018),
flattened with OpenSCAD 2021.01 (`openscad -o model.csg model.scad`) and converted with
`scripts/scad_to_scene.py`, which scales each model into [-1, 1] (`source` keeps the center and
the scale). The repository has no license file, so the converted scenes are not part of this
repository (`.gitignore` leaves them out). To rebuild one, flatten its program and convert it as above,
into `scenes/inversecsg/<name>.json`.
