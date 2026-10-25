#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// Use scalar references at explicitly calculated 64-bit byte addresses. Do not
// index a large runtime array: Vulkan 1.2 array offset math may wrap at 4 GiB.
layout(local_size_x = 128) in;
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer FloatValue { float value; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer InputValue { uint value; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer Word { uint value; };

// Shared ABI with Params in mod.rs.
layout(push_constant) uniform Params {
    uint64_t vectors;
    uint64_t inputs;
    uint64_t scratch;
    uint64_t output_address;
    uint64_t reduce_src; // UINT64_MAX means the initial distance slots
    uint64_t reduce_dst;
    uint64_t taken;      // scratch word holding the slot of the last KNN hit
    uint dims;
    uint rows;
    uint work_count;
    uint candidate_offset;
    uint disk_count;
    uint mode; // 0: region, 1: KNN, 2: region count
    uint reduce_count;
    uint output_index;
    uint row_offset; // first row of the chunk within its segment
    uint row_base;   // added to region output rows
    uint reduce_shift; // 0: reduce every group; else only the group of `taken`
    uint slot_base;    // first slot of a region batch; work_count is its end
} p;
const uint INVALID = 0xffffffffu;
const uint MAX_DISKS = 16u;

uint group_index() {
    return gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
}

uint input_word(uint index) {
    return InputValue(p.inputs + uint64_t(index) * 4ul).value;
}
Word scratch_word(uint64_t index) {
    return Word(p.scratch + index * 4ul);
}
Word output_word(uint64_t index) {
    return Word(p.output_address + index * 4ul);
}
float vector_value(uint id, uint coord) {
    uint64_t index = uint64_t(coord) * uint64_t(p.rows) + uint64_t(id);
    return FloatValue(p.vectors + index * 4ul).value;
}
uint group_count(uint count) {
    return count / 128u + uint(count % 128u != 0u);
}
