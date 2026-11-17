// Push constants of apps/lpm_view.cpp (128 bytes): the mesh is read where the dynamic update
// wrote it, 3 doubles per vertex and 3 indices per triangle, with the tree size of each vertex.
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer F64 { double v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer U32 { uint v[]; };

layout(push_constant) uniform PC {
    mat4 mvp;
    vec4 eye;      // camera position
    F64 static_v;  // static vertices
    F64 new_v;     // new vertices, numbered from nv_static on
    U32 faces;
    uint nv_static;
    uint flags;    // 1: colors by tree size; 2: the edges, as lines; 4: the vertices, as points; 8: large points
    U32 static_t;  // tree size of the cell of each static vertex
    U32 new_t;     // and of each new vertex
} pc;
