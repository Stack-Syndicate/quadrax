#version 450
layout(local_size_x = 64) in;

layout(std430, binding = 0) buffer A { vec3 a[]; };
layout(std430, binding = 1) buffer B { vec3 b[]; };
layout(std430, binding = 2) buffer C { vec3 c[]; };

void main() {
    uint idx = gl_GlobalInvocationID.x;
    c[idx] = a[idx] + b[idx];
}
