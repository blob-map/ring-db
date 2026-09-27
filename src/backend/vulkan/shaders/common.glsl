// Shared ABI with Params in mod.rs. All buffers use 32-bit words; no optional
// integer-width, subgroup, or atomic-float features are required.
layout(local_size_x = 128) in;
layout(set = 0, binding = 0, std430) readonly buffer Vectors { float vectors[]; };
layout(set = 0, binding = 1, std430) readonly buffer Inputs { uint inputs[]; };
layout(set = 0, binding = 2, std430) buffer Scratch { uint scratch[]; };
layout(set = 0, binding = 3, std430) buffer Output { uint output_words[]; };
layout(push_constant) uniform Params {
    uint dims;
    uint rows;
    uint work_count;
    uint candidate_offset;
    uint disk_count;
    uint mode; // 0: range, 1: intersection, 2: KNN
    float lower_sq;
    float upper_sq;
    uint reduce_count;
    uint reduce_src; // UINT_MAX means the initial distance slots
    uint reduce_dst;
    uint output_index;
} p;
const uint INVALID = 0xffffffffu;

uint group_index() {
    return gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
}
